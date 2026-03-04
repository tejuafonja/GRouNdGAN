#!/bin/bash
#SBATCH --job-name=create_grn
#SBATCH --container-image=projects.cispa.saarland:5005\#c01teaf/llm4grn:latest
#SBATCH --partition=vr0
#SBATCH --output=/home/c01teaf/CISPA-home/job_logs/%j-%x.log
#SBATCH --mail-user=tejumade.afonja@cispa.de
#SBATCH --mail-type=FAIL,END
#SBATCH --time 120:00:00

CONFIG_PATH=$1
echo "Running training on ${CONFIG_PATH}"

cd /home/c01teaf/CISPA-az6/llm_tg-2024/GRouNdGAN/

python3.9 src/main.py --config $CONFIG_PATH --create_grn