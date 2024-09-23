import os
import torch
from torch.functional import einsum
import torch.nn as nn
import torch.nn.functional as TF
import numpy as np
from models import find_network_using_name
from models.criterions import GradientLoss, VGGPerceptualLoss
from .base_model import BaseModel
from .networks import init_net
from .embedding import BaseEmbedding
from utils.utils import chunk_batch, find_class_using_name
from utils.visualizer import Visualizee, depth2im
from tqdm import tqdm
import itertools
from options import get_option_setter, str2bool
from .rendering import VolumetricRenderer
from .utils import *
from .nerf_model import NeRFModel, ColorMSELoss, PSNR, L1Loss
import einops
from .nerf_downX_model import GANLoss
from .criterions import SSIM

from .network_codebook import VQCodebook, Codebook

import torchsummary

class RefineModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--refine_network', type=str, default='unetgenerator')
        parser.add_argument('--refine_with_vgg', action='store_true')
        parser.add_argument('--refine_with_l1', action='store_true')
        parser.add_argument('--refine_with_grad', action='store_true')
        parser.add_argument('--refine_with_mse', action='store_true')
        parser.add_argument('--lambda_refine_vgg', type=float, default=1.0)
        parser.add_argument('--lambda_refine_l1', type=float, default=1.0)
        parser.add_argument('--lambda_refine_mse', type=float, default=10.0)
        parser.add_argument('--lambda_refine_grad', type=float, default=1.0)
        parser.add_argument('--refine_as_gan', action='store_true')
        # parser.add_argument('--patch_len', type=int, default=32)
        # parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        parser.add_argument('--embedding_dim', type=int, default=64) # default : 64
        parser.add_argument('--num_embeddings', type=int, default=512) # default : 512
        parser.add_argument('--commitment_cost', type=float, default=0.25)

        parser.add_argument('--network_codebook', action='store_true')
        parser.add_argument('--inference', action='store_true')
        parser.add_argument('--pretrained_codebook', action='store_true')

        opt, _ = parser.parse_known_args()
        for key, network_name in opt.__dict__.items():
            if key.endswith('_network'):
                network_option_setter = get_option_setter(find_class_using_name('models.networks', network_name, type=nn.Module))
                parser = network_option_setter(parser)
        return parser
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.train_loss_names = ['perceptual']

        self.model_names = ['Refine']
        self.netRefine = init_net(find_network_using_name(opt.refine_network)(opt), opt)
        
        # Codebook 초기화
        # self.codebook = Codebook(opt.embedding_dim, opt.num_embeddings).to(self.device)
        self.codebook = VQCodebook(opt.embedding_dim, opt.num_embeddings).to(self.device)
        
        self.models = {
            'R': self.netRefine,
            'Codebook': self.codebook,  # codebook Network 추가
        }
        self.losses = {
            'vgg': VGGPerceptualLoss(opt),
            'mse': ColorMSELoss(opt),
            'l1': L1Loss(opt),
            'psnr': PSNR(opt),
            'grad': GradientLoss(opt),
            'ssim': SSIM(data_range=(-1,1))
        }
        if not self.opt.pretrained_codebook:
            self.train_visual_names = ['sr_gt_refine', 'ref_patches']
        else :
            self.train_visual_names = []
        
        if self.opt.pretrained_codebook:
            self.train_visual_names.append('hr_gt_codebook')
        else:
            self.train_visual_names.append('sr_gt_refine_codebook')
            
        if self.opt.refine_as_gan:
            self.train_loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        elif self.opt.pretrained_codebook:
            self.train_loss_names = ['tot']
            self.val_iter_loss_names = ['tot']
        else:
            self.train_loss_names = ['mse', 'tot']
            self.val_iter_loss_names = ['mse', 'tot', 'psnr_input', 'psnr_refine']
        
        self.val_iter_visual_names = ['sr_gt_refine', 'ref_patches']
        self.val_visual_names = ['sr_refine']
        self.test_visual_names = ['sr_refine', 'sr_imgs_gif', 'refined_imgs_gif']
        
        if self.opt.refine_with_vgg:
            self.train_loss_names.append('vgg')
        if self.opt.refine_with_l1:
            self.train_loss_names.append('l1')
        if self.opt.refine_with_grad:
            self.train_loss_names.append('grad')
        

        if self.isTrain:
            self.optimizer = torch.optim.Adam(self.netRefine.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # codebook optimization
            if self.opt.pretrained_codebook:
                self.optimizer = torch.optim.Adam(self.codebook.parameters(), lr=3e-4, amsgrad=True)
            self.optimizers = [self.optimizer]
            if self.opt.refine_as_gan:
                self.criterionGAN = GANLoss('lsgan').to(self.device)
                self.criterionL1 = torch.nn.L1Loss()
                self.netD = init_net(find_network_using_name('NLayerdiscriminator')(opt), opt)
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
    
    def forward(self):
        # num_ref_patches가 8일 때 그 중 하나의 패치만 선택
        selected_patch_idx = 0  # 0부터 7까지 선택 가능, 여기서는 첫 번째 패치를 선택
        self.data_ref_patch = self.data_ref_patches[:, selected_patch_idx * 3:(selected_patch_idx + 1) * 3, :, :]

        # VQ-VAE Codebook을 이용하여 data_ref_patches를 생성 또는 보강
        if self.opt.network_codebook:
            # for train : HR => codebook => HR 
            cb_hr_patch, loss_hr, x1,x2,x3 = self.codebook(self.data_ref_patch)
            self.cb_hr_patch = cb_hr_patch
            # for inference : SR => codebook => HR 
            #_, loss_lr, _,_,_ = self.codebook(self.data_sr_patch)
        
        if not self.opt.pretrained_codebook: 
            if self.opt.refine_network == 'unetgenerator':
                self.pred = self.netRefine(self.data_sr_patch,x1,x2,x3)
            else:
                self.pred = self.netRefine(self.data_sr_patch, cb_hr_patch)

            self.sr_gt_refine = Visualizee('image', torch.cat([self.data_sr_patch[0], \
                            self.data_gt_patch[0], self.pred[0].detach()], dim=2), \
                            timestamp=True, name='sr_gt_refine', data_format='CHW', range=(-1, 1), img_format='png')
            if self.opt.network_codebook:
                cb_hr_patch_cpu = cb_hr_patch[0].detach().cpu().numpy()
                cb_hr_patch_cpu = torch.tensor(cb_hr_patch_cpu).to(self.device)
                self.sr_gt_refine_codebook = Visualizee('image', torch.cat([self.data_sr_patch[0], \
                                self.data_gt_patch[0], self.pred[0].detach(), cb_hr_patch_cpu], dim=2),\
                                timestamp=True, name='sr_gt_refine_codebook', data_format='CHW', range=(-1, 1), img_format='png')
        
        else:
            self.hr_gt_codebook = Visualizee('image', torch.cat([ self.data_ref_patch[0],cb_hr_patch[0]], dim=2),\
                                timestamp=True, name='sr_gt_refine_codebook', data_format='CHW', range=(-1, 1), img_format='png')

        # Save losses for later uses
        if self.opt.network_codebook:
            self.loss_hr = loss_hr
            #self.loss_lr = loss_lr

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.input, self.pred), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.input, self.gt), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.input, self.pred), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.pred, self.gt) * 100
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters_gan(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def optimize_parameters(self):
        if self.opt.refine_as_gan:
            self.optimize_parameters_gan()
            return

        self.forward()
        self.set_requires_grad([self.netRefine, self.codebook], True) 
        self.optimizer.zero_grad()  
        self.backward()           
        self.optimizer.step()

    def calculate_losses(self):
        self.loss_mse = 0.0
        self.loss_vgg, self.loss_l1 = 0.0, 0.0
        
        if self.opt.pretrained_codebook:
            self.loss_tot = self.loss_hr #+ self.loss_lr
            self.loss_tot += self.losses['mse'](self.cb_hr_patch, self.data_ref_patch) * self.opt.lambda_refine_mse
            with torch.no_grad():
                self.loss_psnr_cb = self.losses['psnr'](self.cb_hr_patch, self.data_ref_patch)

        else:  
            if self.opt.refine_with_vgg:
                self.loss_vgg = self.losses['vgg'](self.pred, self.data_gt_patch) * self.opt.lambda_refine_vgg
            if self.opt.refine_with_l1:
                self.loss_l1 = self.losses['l1'](self.pred, self.data_gt_patch) * self.opt.lambda_refine_l1
            if self.opt.refine_with_mse:
                self.loss_mse = self.losses['mse'](self.pred, self.data_gt_patch) * self.opt.lambda_refine_mse
            self.loss_tot = self.loss_vgg + self.loss_mse + self.loss_l1   
            
            # loss for codebook
            self.loss_tot += self.loss_hr
            self.loss_tot += self.losses['mse'](self.cb_hr_patch, self.data_ref_patch) * self.opt.lambda_refine_mse

            if self.opt.refine_with_grad:
                self.loss_grad = self.losses['grad'](self.pred, self.data_gt_patch) * self.opt.lambda_refine_grad
                self.loss_tot += self.loss_grad

            with torch.no_grad():
                self.loss_psnr_input = self.losses['psnr'](self.data_sr_patch, self.data_gt_patch)
                self.loss_psnr_refine = self.losses['psnr'](self.pred, self.data_gt_patch)
                self.loss_psnr_cb = self.losses['psnr'](self.cb_hr_patch, self.data_ref_patch)
 

    def backward(self):
        self.calculate_losses()
        self.loss_tot.backward()
        # self.calculate_vis()
        # self.loss_tot.backward()
        # self.sr_gt_refine = Visualizee('image', torch.cat([self.data_sr_patch[0], self.data_gt_patch[0], self.pred[0].detach()], dim=2), timestamp=True, name='sr_gt_refine', data_format='CHW', range=(-1, 1), img_format='png')

    def set_input(self, input, need_pack=False):
        pack = lambda x: x.squeeze() if x.shape[0] == 1 else x # N = 1 when val/test/infer
        for name, v in input.items():
            if need_pack:
                v = pack(v)
            setattr(self, f"data_{name}", v.to(self.device))
        # self.real = Visualizee('image', self.data_gan_real_rgbs[0], timestamp=True, name='real', data_format='HWC', range=(0, 1), img_format='png')
        self.ref_patches = Visualizee('image', torch.cat([*self.data_ref_patches[0]], dim=2), timestamp=True, name='ref_patches', data_format='CHW', range=(-1, 1), img_format='png')
        if self.opt.refine_network == 'unetgenerator':
            self.data_ref_patches = self.data_ref_patches.view(self.data_ref_patches.shape[0], -1, self.data_ref_patches.shape[-2], self.data_ref_patches.shape[-1])
    
    def validate_iter(self):
        self.forward()
        if not self.opt.refine_as_gan:
            self.calculate_losses()
        # self.calculate_vis()
        if not self.opt.pretrained_codebook :
            self.sr_gt_refine.name = 'sr_gt_refine_val'
            self.ref_patches.name = 'ref_patches_val'

    def test(self, dataset):
        refined_imgs = []
        sr_imgs = []
        self.sr_refine = []
        sr_psnr = 0.0
        re_psnr = 0.0
        for i, data in enumerate(tqdm(dataset, desc="Testing", total=len(dataset.dataloader))):
            self.set_input(data, need_pack=True)
            self.forward()
            if i % self.opt.test_img_split == 0:
                refine_img = torch.zeros((3, int(self.data_wh[1]), int(self.data_wh[0])))
                sr_img = torch.zeros_like(refine_img)
                gt_img = torch.zeros_like(refine_img)
            for p_idx, patch in enumerate(self.pred):
                loc = [int(self.data_start_locs[p_idx][0]), int(self.data_start_locs[p_idx][1])]
                refine_img[:, loc[1]: loc[1]+self.data_patch_len, loc[0]: loc[0]+self.data_patch_len] = patch
                sr_img[:, loc[1]: loc[1]+self.data_patch_len, loc[0]: loc[0]+self.data_patch_len] = self.data_sr_patch[p_idx]
                gt_img[:, loc[1]: loc[1]+self.data_patch_len, loc[0]: loc[0]+self.data_patch_len] = self.data_gt_patch[p_idx]
            if i % self.opt.test_img_split == self.opt.test_img_split - 1: # finish refining
                refined_imgs.append(refine_img)
                sr_imgs.append(sr_img)
            # sr_img = (sr_img + 1.0) / 2
            # gt_img = (gt_img + 1.0) / 2
            # refine_img = (refine_img + 1.0) / 2
            # print(self.losses['psnr'](sr_img, gt_img), self.losses['psnr'](refine_img, gt_img))
                if i != self.opt.test_img_split - 1:
                    sr_psnr += self.losses['ssim'](sr_img.unsqueeze(0), gt_img.unsqueeze(0))
                    re_psnr += self.losses['ssim'](refine_img.unsqueeze(0), gt_img.unsqueeze(0))

                self.sr_refine.append(
                    Visualizee('image', torch.cat([sr_img, refine_img, gt_img], 2), timestamp=False, name=f'{i//self.opt.test_img_split}-sr-refine', data_format='CHW', range=(-1, 1), img_format='png')
                )
        self.sr_imgs_gif = Visualizee('gif', sr_imgs, timestamp=False, name=f'sr', data_format='CHW', range=(-1, 1))
        self.refined_imgs_gif = Visualizee('gif', refined_imgs, timestamp=False, name=f'refine', data_format='CHW', range=(-1, 1))

    def validate(self, dataset):
        refined_imgs = []
        sr_imgs = []
        self.sr_refine = []
        sr_psnr = 0.0
        re_psnr = 0.0
        for i, data in enumerate(tqdm(dataset, desc="Testing", total=len(dataset.dataloader))):
            self.set_input(data, need_pack=True)
            self.forward()
            if i % self.opt.test_img_split == 0:
                refine_img = torch.zeros((3, int(self.data_wh[1]), int(self.data_wh[0])))
                sr_img = torch.zeros_like(refine_img)
                gt_img = torch.zeros_like(refine_img)
            for p_idx, patch in enumerate(self.pred):
                loc = [int(self.data_start_locs[p_idx][0]), int(self.data_start_locs[p_idx][1])]
                refine_img[:, loc[1]: loc[1]+self.data_patch_len, loc[0]: loc[0]+self.data_patch_len] = patch
                sr_img[:, loc[1]: loc[1]+self.data_patch_len, loc[0]: loc[0]+self.data_patch_len] = self.data_sr_patch[p_idx]
                gt_img[:, loc[1]: loc[1]+self.data_patch_len, loc[0]: loc[0]+self.data_patch_len] = self.data_gt_patch[p_idx]
            
            if i % self.opt.test_img_split == self.opt.test_img_split - 1: # finish refining
                refined_imgs.append(refine_img)
                sr_imgs.append(sr_img)
            # sr_img = (sr_img + 1.0) / 2
            # gt_img = (gt_img + 1.0) / 2
            # refine_img = (refine_img + 1.0) / 2
            # print(self.losses['psnr'](sr_img, gt_img), self.losses['psnr'](refine_img, gt_img))
                if i != self.opt.test_img_split - 1:
                    sr_psnr += self.losses['ssim'](sr_img.unsqueeze(0), gt_img.unsqueeze(0))
                    re_psnr += self.losses['ssim'](refine_img.unsqueeze(0), gt_img.unsqueeze(0))

                self.sr_refine.append(
                    Visualizee('image', torch.cat([sr_img, refine_img, gt_img], 2), timestamp=False, name=f'{i//self.opt.test_img_split}-sr-refine', data_format='CHW', range=(-1, 1), img_format='png')
                )

    def inference(self, dataset):
        # input a whole image
        pass
