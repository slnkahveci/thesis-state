#!/bin/bash

#SBATCH --job-name=grid_search2
#SBATCH --output=grid_search2_output_%j.log
#SBATCH --error=grid_search2_error_%j.log

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

##SBATCH --constraint=node[165-200]

#SBATCH --time=07:30:00



# DON'T USE ARGS

# Load necessary modules
module load python/3.9.19-gpu
module load nvidia/cuda/12.2
module load pytorch/1.9
module avail

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Number of CPU cores: $SLURM_CPUS_PER_TASK"

# Print loaded modules and PATH
module list
echo "PATH is: $PATH"

# Check Python version and location
which python3 || echo "python3 not found in PATH"
python3 --version || echo "python3 command failed"

# Create and activate virtual environment
VENV_DIR="${SLURM_SUBMIT_DIR}/myvenv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created."
fi
source "$VENV_DIR/bin/activate"

# Run your main training script
python3 -u "${SLURM_SUBMIT_DIR}/src/grid_search2.py"

# Deactivate virtual environment
deactivate

echo "Job finished on $(date)"
