#!/bin/bash
#SBATCH --verbose
#SBATCH --nodes=4               # Request 4 nodes
#SBATCH --ntasks-per-node=16    # 16 tasks (MPI ranks) per node => total of 64
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --job-name=bounce64
#SBATCH --mem=100G
#SBATCH --partition=courses

# Load your MPI module if needed (this depends on your cluster)
# module load openmpi
module load OpenMPI/4.1.6
# Typically, under Slurm we can either use srun or mpirun. 
# A common approach with OpenMPI is simply:
#     mpirun -np 64 ./bounce64
#
# Or we can use srun (Slurm's native launcher). One example is:
#     srun --mpi=pmix_v3 -n 64 ./bounce64

# Here is a simple command using srun:
mpirun --oversubscribe -np 64 ./bounce64
