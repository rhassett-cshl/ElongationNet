#!/bin/bash
#$ -cwd
#$ -V
#$ -N elongation_net
#$ -o test_model_output.txt
#$ -e test_model_errors.txt
#$ -S /bin/bash
#$ -l h_rt=1:00:00
#$ -l h_vmem=500G
#$ -l gpu=1
#$ -pe threads 8

python -u main.py --mode=analysis --config_name=cnn_analysis
