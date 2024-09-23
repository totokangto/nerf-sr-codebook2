W=504
H=378
downscale=2
N_importance=64
batch_size=2048
dataset='fern'

data_root='/data/csj000714/data'

python test.py --name llff-${dataset}-${H}x${W}-ni${N_importance}-dp-ds${downscale} \
    --dataset_mode llff_downX --dataset_root ${data_root}/nerf_llff_data/${dataset} \
    --checkpoints_dir ./checkpoints/nerf-sr --summary_dir ./logs/nerf-sr --results_dir ./results/nerf-sr \
    --img_wh ${W} ${H} --batch_size $batch_size \
    --model nerf_downX --N_coarse 64 --N_importance $N_importance \
    --downscale ${downscale} --load_epoch 30 --test_split test