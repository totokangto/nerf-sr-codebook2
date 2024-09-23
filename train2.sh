#!/usr/bin/bash

#SBATCH -J nerf-sr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_eebme_ugrad
#SBATCH -w moana-y3
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

bash scripts/train_llff_refine.sh
# python warp.py
exit 0
