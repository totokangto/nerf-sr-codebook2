# dataset="fern"
# W=504
# H=378
# batch_size=1
# downscale=2
# option="codebook_inference_loss"

# python test_refine.py --name llff-refine-${dataset}-${H}x${W}-ni-dp-ds${downscale}-${option} \
#     --dataset_mode llff_refine --dataset_root /local_datasets/nerf_llff_data/${dataset} \
#     --checkpoints_dir ./checkpoints/nerf-sr-refine/ --summary_dir ./logs/nerf-sr-refine --results_dir ./results/nerf-sr-refine \
#     --img_wh $W $H --batch_size $batch_size \
#     --model refine --test_split test_train \
#     --refine_network unetgenerator \
#     --test_split test --load_epoch 3 \
#     --inference \
#     --network_codebook \
#     --syn_dataroot ./checkpoints/nerf-sr/llff-${dataset}-${H}x${W}-ni64-dp-ds${downscale}/30_test_vis 
#     # train과 동일하게 unetgenerator로 바꿈
#     # --refine_network maxpoolingmodel \

dataset="fern"
W=504
H=378
downscale=2
batch_size=1
option="cb_feature_at_once_100"


python test_refine.py --name llff-refine-$dataset-${option} \
    --dataset_mode llff_refine --dataset_root /local_datasets/nerf_llff_data/${dataset} \
    --checkpoints_dir ./checkpoints/nerf-sr-refine/ --summary_dir ./logs/nerf-sr-refine --results_dir ./results/nerf-sr-refine \
    --img_wh $W $H --batch_size $batch_size \
    --model refine --test_split test_train --load_epoch 3 \
    --syn_dataroot ./checkpoints/nerf-sr/llff-${dataset}-${H}x${W}-ni64-dp-ds${downscale}/30_test_vis \
    --network_codebook 

    # --test_split test --refine_network maxpoolingmodel --load_epoch 3 \
