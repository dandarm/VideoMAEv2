#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:25:00
#SBATCH --error=predjob.err
#SBATCH --output=predjob.out

module load profile/deeplrn
module load cineca-ai/4.3.0
source $HOME/videomae/bin/activate

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12340


mpirun --map-by socket:PE=4 --report-bindings python inference_classification.py \
    --on leonardo \
    --inference_model output/checkpoint-best-lr-again2.pth \
    --csvfile val_manos_w_2400.csv \
    --get_logits
    
#srun --ntasks-per-node=4 \
#python -m torch.distributed.run \
#    --nproc_per_node 4 \
#    --nnodes 4 \
#    specialization.py
# module load openmpi 
