#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=01:59:00
#SBATCH --error=track_from_folder.err
#SBATCH --output=track_from_folder.out

module load profile/deeplrn
module load cineca-ai/4.3.0
source $HOME/videomae/bin/activate

export PYTHONWARNINGS=ignore

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12340

# Aggiorna i path secondo il tuo caso d'uso
mpirun --map-by socket:PE=4 --report-bindings python track_from_folder.py \
    --on leonardo \
    --input_dir $FAST/airmassRGB/supervised_tiles2track \
    --output_dir $FAST/airmassRGB/supervised_tiles_tracked \
    --model_path $FAST/checkpoint-tracking-best.pth \
    --manos_file medicane_data_input/medicanes_new_windows.csv \
    --make_video
