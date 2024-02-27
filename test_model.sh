#!/bin/bash
#$ -cwd
#$ -V
#$ -N elongation_net
#$ -o output_save.txt
#$ -e errors_save.txt
#$ -S /bin/bash
#$ -l h_rt=1:00:00
#$ -l h_vmem=100G
#$ -l gpu=0
#$ -pe threads 8

python test_model.py
