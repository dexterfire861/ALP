#!/bin/bash
#SBATCH --job-name=docling
#SBATCH --mail-user=smaley@ufl.edu
#SBATCH --mail-type=ALL
#SBATCH --output docling-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gpus=a100:1
#SBATCH --account=rcstudents
#SBATCH --qos=rcstudents

date;hostname;pwd
module load conda
conda activate alp

python docling_run.py

date;hostname;pwd

