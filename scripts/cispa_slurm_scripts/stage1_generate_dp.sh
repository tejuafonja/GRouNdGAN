#!/bin/bash

for DATASET in COVID_Haniffa21-GGpp-covid-privacy
do
    for CV in 1000
    do
        for exp in eps10
        do
        CONFIG_PATH=configs/$DATASET/CrossVal_$CV/${exp}.cfg
        sbatch /home/c01teaf/CISPA-az6/llm_tg-2024/GRouNdGAN/scripts/cispa_slurm_scripts/run_generate.sh $CONFIG_PATH
        done
    done
done

