#!/bin/bash

#SBATCH --partition=all-gpu

#SBATCH --time=8:00:00

#SBATCH --node=1

#SBATCH --ntasks-per-node=1

#SBATCH --gres:gpu:1

#SBATCH --job-name=segtry

#SBATCH --output="segmentation.%j.%N.out"

#SBATCH --mail-type=ALL
#SBATCH --mail-user=lid315@lehigh.edu

cd /home/lid315/keras

module load anaconda/mldl

srun -n1 -c2 ./modelslurm.py > output.out