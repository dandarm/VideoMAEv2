#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=05:58:00
#SBATCH --error=myJob.err
#SBATCH --output=tracking_job.out

module load profile/deeplrn
module load cineca-ai/4.3.0
source $HOME/videomae/bin/activate

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12340


mpirun --map-by socket:PE=4 --report-bindings python tracking.py --on leonardo

#srun --ntasks-per-node=4 \
#python -m torch.distributed.run \
#    --nproc_per_node 4 \
#    --nnodes 4 \
#    specialization.py
# module load openmpi 
