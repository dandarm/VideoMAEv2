#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --time=02:25:00
#SBATCH --error=preprocess.err
#SBATCH --output=preprocess.out

module load profile/deeplrn
module load cineca-ai/4.3.0
source $HOME/videomae/bin/activate

export PYTHONWARNINGS=ignore

srun python predict_from_folder.py \
    --on leonardo \
    --input_dir ../fromgcloud/2023 \
    --output_dir ../airmassRGB/supervised \
    --model_path $FAST/checkpoint-best-lr-again2.pth \
    --manos_file medicane_data_input/medicanes_new_windows.csv

