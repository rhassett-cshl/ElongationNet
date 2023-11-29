#!/bin/bash

#$ -N ElongationNetJob     # Job name
#$ -l h_rt=3:00:00        # Specify the hard time limit for the job
#$ -l s_rt=2:00:00        # Specify the soft time limit for the job
#$ -pe threads 10              # Request 10 cores
#$ -l h_vmem=250M          # Request 512MB of memory per core
#$ -l gpu=2                # Request 2 GPUs
#$ -cwd                    # Run job from the current directory
#$ -V                      # Export environment variables
#$ -o output.txt           # Name of the stdout, using the job name
#$ -e error.txt            # Name of the stderr, using the job name

# Your commands go here
conda activate cnn-motif
python ./elongation_net_chr22.py 
