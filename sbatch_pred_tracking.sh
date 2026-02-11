#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --error=predjob.err
#SBATCH --output=predjob.out

module load profile/deeplrn
module load cineca-ai/4.3.0
source $HOME/videomae/bin/activate

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12340

export PYTHONWARNINGS=ignore

mpirun --map-by socket:PE=4 --report-bindings python inference_tracking.py \
    --on leonardo \
    --inference_model $FAST/checkpoint-tracking-best.pth \
    --csvfile test_tracking_selezionati.csv \
    
    
#srun --ntasks-per-node=4 \
#python -m torch.distributed.run \
#    --nproc_per_node 4 \
#    --nnodes 4 \
#    specialization.py
# module load openmpi 
