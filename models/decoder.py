
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.residual import ResidualStack

class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2
        '''
        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),

            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim ,
                               kernel_size=kernel-1, stride=stride-1, padding=1),

            nn.ReLU(),
            nn.ConvTranspose2d(h_dim, h_dim//2, 
                               kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim//2, 3, kernel_size=kernel,
                               stride=stride, padding=1)
        )
        '''
        self.up_1 =nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, h_dim, kernel_size=1, stride=1),
                ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        )
 
        self.up_2 = nn.Sequential(
            nn.ConvTranspose2d(h_dim, h_dim ,
                               kernel_size=kernel-1, stride=stride-1, padding=1)
        )

        self.up_3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim, h_dim//2, kernel_size=kernel,
                               stride=stride, padding=1)
        )

        self.up_4 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim//2, 3, kernel_size=kernel,
                               stride=stride, padding=1)
        )

    def forward(self, x):
        x1 = self.up_1(x)
        x2 = self.up_2(x1)
        x3 = self.up_3(x2)
        x4 = self.up_4(x3)
        return x1,x2,x3,x4