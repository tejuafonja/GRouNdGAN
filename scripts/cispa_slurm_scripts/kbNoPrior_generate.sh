#!/bin/bash
# PBMC_CTL PBMC BoneMarrow
# GPT4.1 GPT4 GRNB2_Bottom GRNB2_Random GRNB2 DeepSEM Llama PIDC
for GRN in PIDC DeepSEM
do
    for DATASET in PBMC
    do
        for FREQ in KB_Sapien
        do
            for CV in 1000
            do
                CONFIG_PATH=configs/$DATASET/CrossVal_$CV/KB_No_Prior/freq_${FREQ}/Seed1/$GRN.cfg
                sbatch /home/c01teaf/CISPA-az6/llm_tg-2024/GRouNdGAN/scripts/cispa_slurm_scripts/run_generate.sh $CONFIG_PATH
            done
        done
    done
done