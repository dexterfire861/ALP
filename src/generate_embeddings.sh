#!/bin/bash
#SBATCH --job-name=generate_embeddings
#SBATCH --mail-user=smaley@ufl.edu
#SBATCH --mail-type=ALL
#SBATCH --output generate_embeddings-%j.txt
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
conda activate alp-env

python generate_embeddings.py

date;hostname;pwd

