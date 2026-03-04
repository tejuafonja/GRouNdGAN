#!/bin/bash

for DATASET in COVID_Haniffa21-GGpp-healthy COVID_Haniffa21-GGpp-covid
do
    for CV in 1000
    do
        CONFIG_PATH=configs/$DATASET/CrossVal_$CV/KB_Sapien/freq_None/Seed1/GRNB2.cfg
        sbatch /home/c01teaf/CISPA-az6/llm_tg-2024/GRouNdGAN/scripts/cispa_slurm_scripts/run_create_grn.sh $CONFIG_PATH
    done
done