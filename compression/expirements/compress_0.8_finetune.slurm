#!/bin/bash

#SBATCH --output=compress_0.8_finetune_%J.out
#SBATCH --error=compress_0.8_finetune_%J.err
#SBATCH --job-name=compress_0.8_finetune

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=8192
#SBATCH --mail-type=BEGIN,END,FAIL   
#SBATCH --mail-user=danielzgsilva@knights.ucf.edu 

echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

source activate /home/dsilva/my-envs/MOT

echo 'Beginning script'

python finetune.py --compress_factor 0.8 --finetune_model_name compress_0.8

cd ../

echo "Ending script..."

