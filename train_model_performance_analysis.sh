#!/bin/bash
#$ -cwd
#$ -V
#$ -N train_model_performance_analysis_k562
#$ -o output_perf_k562.txt
#$ -e errors_perf_k562.txt
#$ -S /bin/bash
#$ -l h_rt=1:00:00
#$ -l h_vmem=500G
#$ -l gpu=0
#$ -pe threads 8

python -u main.py --mode=performance_analysis --config_name=cnn_performance_analysis
