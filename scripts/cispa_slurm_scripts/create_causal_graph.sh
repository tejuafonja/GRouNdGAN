#!/bin/bash
#SBATCH --job-name=create_causal_graph_llm4grn
#SBATCH --container-image=projects.cispa.saarland:5005\#c01teaf/llm4grn:latest
#SBATCH --gres=cpu
#SBATCH --partition=vr0
#SBATCH --output=/home/c01teaf/CISPA-home/job_logs/%j-%x.log
#SBATCH --mail-user=tejumade.afonja@cispa.de
#SBATCH --mail-type=FAIL,END
#SBATCH --time 120:00:00
#SBATCH --exclude=xe8545-a100-09,xe8545-a100-29,xe8545-a100-30

CONFIG_PATH=$1
echo "Running training on ${CONFIG_PATH}"

cd /home/c01teaf/CISPA-az6/llm_tg-2024/GRouNdGAN/

GRN=GRNB2
DATASET=COVID_Haniffa21-GGpp-healthy
CV=1000
CONFIG_PATH=configs/$DATASET/CrossVal_$CV/KB_Sapien/freq_None/Seed1/$GRN.cfg

python3.9 src/main.py --config $CONFIG_PATH --create_grn