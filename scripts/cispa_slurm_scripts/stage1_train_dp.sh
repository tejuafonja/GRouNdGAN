#!/bin/bash

# for DATASET in COVID_Haniffa21-GGpp-covid-privacy
# do
#     for exp in gpr20_ms15 gpr20_ms20 gpr20_ms25
#     do
#         CONFIG_PATH=configs/$DATASET/CrossVal_1000/eps30/round10/${exp}.cfg
#         sbatch /home/c01teaf/CISPA-az6/llm_tg-2024/GRouNdGAN/scripts/cispa_slurm_scripts/run_train.sh $CONFIG_PATH
#     done
# done

# for DATASET in COVID_Haniffa21-GGpp-covid-privacy
# do
#     for exp in eps20 eps30
#     do
#         CONFIG_PATH=configs/$DATASET/CrossVal_1000/${exp}/round11/gpr20_ms50.cfg
#         sbatch /home/c01teaf/CISPA-az6/llm_tg-2024/GRouNdGAN/scripts/cispa_slurm_scripts/run_train.sh $CONFIG_PATH
#     done
# done


# for DATASET in COVID_Haniffa21-GGpp-covid-privacy
# do
#     for exp in eps20 eps30
#     do
#         CONFIG_PATH=configs/$DATASET/CrossVal_1000/${exp}/round12/gpr20_ms50.cfg
#         sbatch /home/c01teaf/CISPA-az6/llm_tg-2024/GRouNdGAN/scripts/cispa_slurm_scripts/run_train.sh $CONFIG_PATH
#     done
# done


# for DATASET in COVID_Haniffa21-GGpp-covid-privacy
# do
#     for exp in eps20 eps30
#     do
#         CONFIG_PATH=configs/$DATASET/CrossVal_1000/${exp}/round13/gpr20_ms50.cfg
#         sbatch /home/c01teaf/CISPA-az6/llm_tg-2024/GRouNdGAN/scripts/cispa_slurm_scripts/run_train.sh $CONFIG_PATH
#     done
# done


# for DATASET in COVID_Haniffa21-GGpp-covid-privacy
# do
#     for exp in eps20 eps30
#     do
#         CONFIG_PATH=configs/$DATASET/CrossVal_1000/${exp}/round14/gpr20_ms50.cfg
#         sbatch /home/c01teaf/CISPA-az6/llm_tg-2024/GRouNdGAN/scripts/cispa_slurm_scripts/run_train.sh $CONFIG_PATH
#     done
# done

#  
#  eps15 eps20 eps50 eps100 eps1000 nodp
# COVID_Haniffa21-GGpp-covid-privacy 
# COVID_Haniffa21-GGpp-healthy-privacy

for exp in eps10 eps20 eps40 eps50 eps100 eps1000
do
    for DATASET in COVID_Haniffa21-GGpp-covid-privacy COVID_Haniffa21-GGpp-healthy-privacy
    do
        CONFIG_PATH=configs/$DATASET/CrossVal_1000/${exp}.cfg
        sbatch /home/c01teaf/CISPA-az6/llm_tg-2024/GRouNdGAN/scripts/cispa_slurm_scripts/run_train.sh $CONFIG_PATH
    done
done

