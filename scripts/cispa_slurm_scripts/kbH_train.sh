#!/bin/bash
# GPT4.1 GPT4 GRNB2_Bottom GRNB2_Random GRNB2 DeepSEM Llama
for GRN in GRNB2_Bottom
do
    for DATASET in PBMC
    do
        for CV in 1000
        do
            CONFIG_PATH=configs/$DATASET/CrossVal_$CV/KB_Sapien/freq_None/Seed1/$GRN.cfg
            sbatch /home/c01teaf/CISPA-az6/llm_tg-2024/GRouNdGAN/scripts/cispa_slurm_scripts/run_train.sh $CONFIG_PATH
        done
    done
done