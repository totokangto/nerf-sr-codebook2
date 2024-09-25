dataset="fern"
W=504
H=378
accelerator="dp"
downscale=2
batch_size=32
option="cb_feature_at_once_64_100"


python train_refine.py --name llff-refine-$dataset-${option} --accelerator $accelerator \
    --dataset_mode llff_refine --dataset_root /local_datasets/nerf_llff_data/${dataset} \
    --checkpoints_dir ./checkpoints/nerf-sr-refine --summary_dir ./logs/nerf-sr-refine \
    --img_wh $W $H --batch_size $batch_size \
    --n_epochs 100 --n_epochs_decay 0 \
    --print_freq 7500 --vis_freq 7500 --val_freq 15000 --save_epoch_freq 20 --val_epoch_freq 5 \
    --model refine \
    --lr_policy exp --lr 5e-4 --lr_final 5e-6 \
    --syn_dataroot ./checkpoints/nerf-sr/llff-${dataset}-${H}x${W}-ni64-dp-ds${downscale}/30_val_vis \
    --refine_with_l1 --network_codebook 
