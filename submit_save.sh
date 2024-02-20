#!/bin/bash
#$ -cwd
#$ -V
#$ -N elongation_net
#$ -o output_save.txt
#$ -e errors_save.txt
#$ -S /bin/bash
#$ -l h_rt=10:00:00
#$ -l h_vmem=500G
#$ -l gpu=1
#$ -pe threads 8

python -u main.py --mode=save_results --config_name=ep_seq_linear_1
