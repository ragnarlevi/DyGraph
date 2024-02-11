#!/bin/bash --login
#SBATCH --job-name=run_prefecture
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=120:00:00

echo "Hi"

# Load the R module
flight env activate gridware
module load apps/anaconda3/2023.03

RESULTS_DIR="$(pwd)"
echo "Your results will be stored in: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

echo "Running"
conda init bash
conda activate aflac
# Run the python script
python CaseStudyETF.py

echo "Finished"
