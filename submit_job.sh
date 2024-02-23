#!/bin/bash
#$ -cwd
#$ -V
#$ -N elongation_net
#$ -o output.txt
#$ -e errors.txt
#$ -S /bin/bash
#$ -l h_rt=10:00:00
#$ -l h_vmem=500G
#$ -l gpu=1
#$ -pe threads 8

python -u main.py --mode=train --config_name=cnn_2
