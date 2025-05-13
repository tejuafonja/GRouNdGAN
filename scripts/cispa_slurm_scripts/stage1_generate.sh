#!/bin/bash

for DATASET in BoneMarrow PBMC PBMC_CTL
do
    for CV in 1000
    do
        CONFIG_PATH=configs/$DATASET/CrossVal_$CV/Stage1.cfg
        sbatch /home/c01teaf/CISPA-az6/llm_tg-2024/GRouNdGAN/scripts/cispa_slurm_scripts/run_generate_cc.sh $CONFIG_PATH
    done
done