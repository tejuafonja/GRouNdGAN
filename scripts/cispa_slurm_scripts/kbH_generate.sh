#!/bin/bash

# GPT4.1 DeepSEM GPT4 GRNB2_Random GRNB2 Llama GRNB2_Bottom PIDC

for GRN in GRNB2
do
    for DATASET in COVID_Haniffa21
    do
        for CV in 1000
        do
            CONFIG_PATH=configs/$DATASET/CrossVal_$CV/KB_Sapien/freq_None/Seed1/$GRN.cfg
            sbatch /home/c01teaf/CISPA-az6/llm_tg-2024/GRouNdGAN/scripts/cispa_slurm_scripts/run_generate.sh $CONFIG_PATH
        done
    done
done