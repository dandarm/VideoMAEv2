#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=01:59:00
#SBATCH --error=predjob.err
#SBATCH --output=predjob.out

module load profile/deeplrn
module load cineca-ai/4.3.0
source $HOME/videomae/bin/activate

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12340

# Aggiorna i path secondo il tuo caso d'uso
mpirun --map-by socket:PE=4 --report-bindings python predict_from_folder.py \
    --on leonardo \
    --input_dir $FAST/Medicanes_Data/from_gcloud/2023 \
    --output_dir $FAST/airmass \
    --model_path $FAST/checkpoint-best-lr-again2.pth \
    --manos_file medicane_data_input/medicanes_new_windows.csv
