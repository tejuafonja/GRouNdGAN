#!/bin/bash
# PBMC_CTL PBMC BoneMarrow
# GPT4.1 GPT4 GRNB2_Bottom GRNB2_Random GRNB2 DeepSEM Llama PIDC
# 1_hacky 1
for GRN in PIDC
do
    for DATASET in PBMC
    do
        for FREQ in 1
        do
            for CV in 1000
            do
                CONFIG_PATH=configs/$DATASET/CrossVal_$CV/KB_GPT4.1/freq_${FREQ}/Seed1/$GRN.cfg
                sbatch /home/c01teaf/CISPA-az6/llm_tg-2024/GRouNdGAN/scripts/cispa_slurm_scripts/run_train.sh $CONFIG_PATH
            done
        done
    done
done