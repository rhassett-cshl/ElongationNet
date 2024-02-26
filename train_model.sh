#!/bin/bash
#$ -cwd
#$ -V
#$ -N elongation_net
#$ -o train_model_output.txt
#$ -e train_model_errors.txt
#$ -S /bin/bash
#$ -l h_rt=0:30:00
#$ -l h_vmem=250G
#$ -l gpu=1
#$ -pe threads 7

python -u main.py --mode=train --config_name=cnn_analysis
