#!/bin/bash

#SBATCH --verbose
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --job-name=DavesJob
#SBATCH --mem=100G
#SBATCH --partition=courses

# Load the MPI module if needed (depends on your system)
# module load mpi
module load OpenMPI/4.1.6

# (Optional) Print the environment or diagnostic info
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on $SLURM_NNODES nodes with $SLURM_NTASKS_PER_NODE tasks per node"
echo "Total tasks: $SLURM_NTASKS"


# Run the parallel histogram program:
#   - Replace ~/MPI/parallel_histogram with the actual path to your executable
#   - Pass the number of bins (e.g., 128) on the command line
# srun ~/parallel_histogram 128

# If you want to run multiple experiments in one script, you can add more lines:
# srun ~/MPI/parallel_histogram 32
# srun ~/MPI/parallel_histogram 256

mpirun -np $SLURM_NTASKS ~/parallel_histogram 32
