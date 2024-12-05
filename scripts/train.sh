#!/bin/bash

#SBATCH --job-name=groundgan
#SBATCH --nodes=1 # How many nodes? 
#SBATCH -A hai_steerllms # Who pays for it?
#SBATCH --partition booster 
#SBATCH --gres=gpu:4 # requests 4 gpus on the node
#SBATCH --time=24:00:00 
#SBATCH -o output-%x.txt 
#SBATCH -e error-%x.txt
#SBATCH --mail-user=tejumade.afonja@cispa.de
#SBATCH --mail-type=ALL
# Where does the code run?
# Required for legacy reasons
# How long?
# ntasks-per-node=4

nvidia-smi
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

PROJECT_DIR=/p/project1/hai_steerllms/GRouNdGAN
LOG_DIR=${PROJECT_DIR}/scripts/logs
source ${PROJECT_DIR}/activate.sh

# SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"
# source /p/home/jusers/afonja1/juwels/.bashrc
# source /p/project1/hai_steerllms/GRouNdGAN/venv/bin/activate
# echo Insanity
# export PYTHONPATH="$(echo /p/project1/hai_steerllms/GRouNdGAN/venv/lib/python*/site-packages):${PYTHONPATH}"
# echo `python -c "print('hello')"`

echo $PATH
echo $PYTHONPATH


cd ${PROJECT_DIR}

mkdir ${LOG_DIR}
# python src/main.py --config ${PROJECT_DIR}/configs/causal_gan.cfg --preprocess --create_grn
# python src/main.py --config ${PROJECT_DIR}/configs/causal_gan.cfg --create_grn
# tensorboard --logdir="${PROJECT_DIR}/results/GRouNdGAN/TensorBoard" --host 0.0.0.0 --load_fast false &

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-PBMC.txt" -e "$LOG_DIR/error-stage1-PBMC.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/causal_gan_Stage1.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-BM.txt" -e "$LOG_DIR/error-stage1-BM.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/causal_gan_Stage1.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-CTL.txt" -e "$LOG_DIR/error-stage1-CTL.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/causal_gan_Stage1.cfg --train&
# wait



# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-CTL-GT-TF15.txt" -e "$LOG_DIR/error-CTL-GT-TF15.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/EXP_TF_k/causal_gan_TF15.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-CTL-GT.txt" -e "$LOG_DIR/error-CTL-GT.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/causal_gan_GT.cfg --train&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-CTL-GT-BottomTF.txt" -e "$LOG_DIR/error-CTL-GT-BottomTF.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/causal_gan_GT.cfg --train&
# wait


# 
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-PBMC-GT.txt" -e "$LOG_DIR/error-PBMC-GT.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/causal_gan_GT.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-BM-GT.txt" -e "$LOG_DIR/error-BM-GT.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/causal_gan_GT.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-CTL-GT.txt" -e "$LOG_DIR/error-CTL-GT.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/causal_gan_GT.cfg --train&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-PBMC-Rand.txt" -e "$LOG_DIR/error-PBMC-Rand.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/causal_gan_Random.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-CTL-Rand.txt" -e "$LOG_DIR/error-CTL-Rand.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/causal_gan_Random.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-BM-Rand.txt" -e "$LOG_DIR/error-BM-Rand.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/causal_gan_Random.cfg --train&
# wait


# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-PBMC-GPT.txt" -e "$LOG_DIR/error-PBMC-GPT.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/causal_gan_GPT.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-CTL-GPT.txt" -e "$LOG_DIR/error-CTL-GPT.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/causal_gan_GPT.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-BM-GPT.txt" -e "$LOG_DIR/error-BM-GPT.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/causal_gan_GPT.cfg --train&
# wait




# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-PBMC-GT.txt" -e "$LOG_DIR/error-PBMC-GT.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/causal_gan_GT.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-PBMC-GPT.txt" -e "$LOG_DIR/error-PBMC-GPT.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/causal_gan_GPT.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-PBMC-Random.txt" -e "$LOG_DIR/error-PBMC-Random.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/causal_gan_Random.cfg --train&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-BM-GT.txt" -e "$LOG_DIR/error-BM-GT.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/causal_gan_GT.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-BM-GPT.txt" -e "$LOG_DIR/error-BM-GPT.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/causal_gan_GPT.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-BM-Random.txt" -e "$LOG_DIR/error-BM-Random.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/causal_gan_Random.cfg --train&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-CTL-GT.txt" -e "$LOG_DIR/error-CTL-GT.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/causal_gan_GT.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-CTL-GPT.txt" -e "$LOG_DIR/error-CTL-GPT.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/causal_gan_GPT.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-CTL-Random.txt" -e "$LOG_DIR/error-CTL-Random.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/causal_gan_Random.cfg --train&
# wait


# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-1.txt" -e "$LOG_DIR/error-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/causal_gan_GT.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-test-1.txt" -e "$LOG_DIR/error-test-1.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/test_1.cfg --train --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-test-2.txt" -e "$LOG_DIR/error-test-2.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/test_2.cfg --train --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-LU.txt" -e "$LOG_DIR/error-stage1-LU.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/causal_gan_Stage1.cfg --train&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-BM-GPT.txt" -e "$LOG_DIR/error-BM-GT.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/causal_gan_GPT.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-LU-GT.txt" -e "$LOG_DIR/error-LU-GT.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/causal_gan_GT.cfg --train&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-SL-LU.txt" -e "$LOG_DIR/error-stage1-SL-LU.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/synthetic_loop/stage1/iter0.cfg --train --generate_cc&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-SL-LU.txt" -e "$LOG_DIR/error-stage1-SL-LU.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/synthetic_loop/stage1/iter1.cfg --train --generate_cc&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-SL-LU.txt" -e "$LOG_DIR/error-stage1-SL-LU.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/synthetic_loop/stage1/iter2.cfg --train --generate_cc&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-SL-LU.txt" -e "$LOG_DIR/error-stage1-SL-LU.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/synthetic_loop/stage1/iter3.cfg --train --generate_cc&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-SL-LU.txt" -e "$LOG_DIR/error-stage1-SL-LU.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/synthetic_loop/stage1/iter4.cfg --train --generate_cc&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-SL-LU.txt" -e "$LOG_DIR/error-stage1-SL-LU.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/synthetic_loop/stage1/iter5.cfg --train --generate_cc&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-SL-LU.txt" -e "$LOG_DIR/error-stage1-SL-LU.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/synthetic_loop/stage2/iter0.cfg --train --generate
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-SL-LU.txt" -e "$LOG_DIR/error-stage1-SL-LU.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/synthetic_loop/stage2/iter1.cfg --preprocess --train --generate
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-SL-LU.txt" -e "$LOG_DIR/error-stage1-SL-LU.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/synthetic_loop/stage2/iter2.cfg --train --generate
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-SL-LU.txt" -e "$LOG_DIR/error-stage1-SL-LU.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/synthetic_loop/stage2/iter3.cfg --train --generate
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-SL-LU.txt" -e "$LOG_DIR/error-stage1-SL-LU.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/synthetic_loop/stage2/iter4.cfg --preprocess
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-SL-LU.txt" -e "$LOG_DIR/error-stage1-SL-LU.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/synthetic_loop/stage2/iter4.cfg --train --generate
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-SL-LU.txt" -e "$LOG_DIR/error-stage1-SL-LU.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/synthetic_loop/stage2/iter5.cfg --preprocess
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-SL-LU.txt" -e "$LOG_DIR/error-stage1-SL-LU.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/synthetic_loop/stage2/iter5.cfg --train --generate
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-SL-LU.txt" -e "$LOG_DIR/error-stage1-SL-LU.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/synthetic_loop/stage2/iter6.cfg --preprocess
# python src/main.py --config configs/LinearUniform/synthetic_loop/stage1/iter1.cfg --preprocess


# ICLR Experiments
# Stage1
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-CTL-1.txt" -e "$LOG_DIR/error-stage1-CTL-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/Stage1.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-CTL-2.txt" -e "$LOG_DIR/error-stage1-CTL-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/Stage1.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-BM-3.txt" -e "$LOG_DIR/error-stage1-BM-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/Stage1.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-BM-4.txt" -e "$LOG_DIR/error-stage1-BM-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/Stage1.cfg --train&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-LU-1.txt" -e "$LOG_DIR/error-stage1-LU-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/CrossVal_1000/Stage1.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-LU-2.txt" -e "$LOG_DIR/error-stage1-LU-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/CrossVal_2000/Stage1.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-BM-3.txt" -e "$LOG_DIR/error-stage1-BM-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/Stage1.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-BM-4.txt" -e "$LOG_DIR/error-stage1-BM-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/Stage1.cfg --train&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-PBMC-1.txt" -e "$LOG_DIR/error-stage1-PBMC-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/Stage1.cfg --generate_cc&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-PBMC-2.txt" -e "$LOG_DIR/error-stage1-PBMC-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/Stage1.cfg --generate_cc&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-PBMC-3.txt" -e "$LOG_DIR/error-stage1-PBMC-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_3000/Stage1.cfg --generate_cc&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-PBMC-4.txt" -e "$LOG_DIR/error-stage1-PBMC-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_4000/Stage1.cfg --generate_cc&
# wait

# Stage2
# KB_Sapien
# TAG=PBMC_Sapien
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Sapien/freq_None/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Sapien/freq_None/Seed1/Random.cfg --train&
# # srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --train&
# wait

# TAG=PBMC_Sapien_2
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Sapien/freq_None/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Sapien/freq_None/Seed1/Random.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --train&
# wait

# -------PBMC KB GPT and Random
# TAG=PBMC_freq1+KBRandom
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_GPT/freq_1/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_GPT/freq_1/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_GPT/freq_1/Seed1/Random.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Random/freq_None/Seed1/Random.cfg --train&
# wait

# TAG=PBMC_freq2
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_GPT/freq_2/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_GPT/freq_2/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_GPT/freq_2/Seed1/Random.cfg --train&
# wait

# TAG=PBMC_freq1_2000+KBRandom
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_1/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_1/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_1/Seed1/Random.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Random/freq_None/Seed1/Random.cfg --train&
# wait

# TAG=PBMC_freq2_2000
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_2/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_2/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_2/Seed1/Random.cfg --train&
# wait
# -------

# -------PBMC KB Llama and KB GPT freq2 GRNB2
# TAG=PBMC_freq1+
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/freq_1/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/freq_1/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/freq_1/Seed1/Random.cfg --train&
# # srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_2/Seed1/GRNB2.cfg --train &
# wait

# -------PBMC KB Llama and KB GPT freq2 GRNB2
# TAG=PBMC_freq2
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/freq_2/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/freq_2/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/freq_2/Seed1/Random.cfg --train&
# # srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_2/Seed1/GRNB2.cfg --train &
# wait

# TAG=PBMC_Llama
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Llama/no_kb_gpt/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/no_kb_gpt/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Llama/no_kb_human/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/no_kb_human/Seed1/GRNB2.cfg --train &
# wait

# TAG=PBMC_freq2
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Llama/freq_2/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Llama/freq_2/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Llama/freq_2/Seed1/Random.cfg --train&
# wait

# TAG=PBMC_freq1_2000
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/freq_1/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/freq_1/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/freq_1/Seed1/Random.cfg --train&
# wait

# TAG=PBMC_freq2_2000
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/freq_2/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/freq_2/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/freq_2/Seed1/Random.cfg --train&
# wait
# -------

# -------BM KB GPT and Random
# TAG=BM_freq1+KBRandom
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_GPT/freq_1/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_GPT/freq_1/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_GPT/freq_1/Seed1/Random.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_Random/freq_None/Seed1/Random.cfg --train&
# wait

# TAG=BM_freq2
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_GPT/freq_2/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_GPT/freq_2/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_GPT/freq_2/Seed1/Random.cfg --train&
# wait

# TAG=BM_freq1_2000+KBRandom
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_GPT/freq_1/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_GPT/freq_1/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_GPT/freq_1/Seed1/Random.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Random/freq_None/Seed1/Random.cfg --train&
# wait

# TAG=BM_freq2_2000
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_GPT/freq_2/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_GPT/freq_2/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_GPT/freq_2/Seed1/Random.cfg --train&
# wait
# -------


# -------CTL KB GPT and Random
# TAG=CTL_freq1+KBRandom
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/KB_GPT/freq_1/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/KB_GPT/freq_1/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/KB_GPT/freq_1/Seed1/Random.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/KB_Random/freq_None/Seed1/Random.cfg --train&
# wait

# TAG=CTL_freq2
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/KB_GPT/freq_2/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/KB_GPT/freq_2/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/KB_GPT/freq_2/Seed1/Random.cfg --train&
# wait

# TAG=CTL_freq1_2000+KBRandom
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_GPT/freq_1/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_GPT/freq_1/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_GPT/freq_1/Seed1/Random.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_Random/freq_None/Seed1/Random.cfg --train&
# wait

# TAG=CTL_freq2_2000
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_GPT/freq_2/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_GPT/freq_2/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_GPT/freq_2/Seed1/Random.cfg --train&
# wait
# -------

# -------KB Random
# TAG=PBMC_freq1+KBRandom
# # srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Random/freq_None/Seed1/Random.cfg --train&
# # srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/KB_Random/freq_None/Seed1/Random.cfg --train&
# # srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_Random/freq_None/Seed1/Random.cfg --train&
# # wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Random/freq_None/Seed1/Random.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_Random/freq_None/Seed1/Random.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Random/freq_None/Seed1/Random.cfg --train&
# wait
# # # -------

# -------KB Sapien
# TAG=PBMC_Llama
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Sapien/freq_None/Seed1/Llama.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Sapien/freq_None/Seed1/Llama.cfg --train&
# wait
# # -------


# TAG=PBMC_2_freq1_and_2_GRNB2+GPT4
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_1/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_2/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_1/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_2/Seed1/GPT4.cfg --train&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Sapien/freq_None/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Sapien/freq_None/Seed1/Random.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_4000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --generate&
# wait

# TAG=PBMC_Sapien_Random
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Sapien/freq_None/Seed1/Random.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Sapien/freq_None/Seed1/Random.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_3000/KB_Sapien/freq_None/Seed1/Random.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_4000/KB_Sapien/freq_None/Seed1/Random.cfg --train&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Sapien/freq_None/Seed1/Random.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Sapien/freq_None/Seed1/Random.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_3000/KB_Sapien/freq_None/Seed1/Random.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_4000/KB_Sapien/freq_None/Seed1/Random.cfg --generate&
# wait

# TAG=PBMC_Sapien_GPT4
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Sapien/freq_None/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Sapien/freq_None/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_3000/KB_Sapien/freq_None/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_4000/KB_Sapien/freq_None/Seed1/GPT4.cfg --train&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Sapien/freq_None/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Sapien/freq_None/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_3000/KB_Sapien/freq_None/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_4000/KB_Sapien/freq_None/Seed1/GPT4.cfg --generate&
# wait

# TAG=CTL
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_2/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_2/Seed1/GPT4.cfg --train&
# wait

# TAG=BM
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Sapien/freq_None/Seed1/Random.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/CrossVal_2000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_Sapien/freq_None/Seed1/Random.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --train&
# wait

# TAG=LU
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/CrossVal_1000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/CrossVal_2000/KB_Sapien/freq_None/Seed1/GRNB2.cfg--train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Sapien/freq_None/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Sapien/freq_None/Seed1/GRNB2.cfg--train&
# wait

# TAG=CTL+BM
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/KB_Sapien/freq_None/Seed1/Random.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_Sapien/freq_None/Seed1/Random.cfg--train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_Sapien/freq_None/Seed1/Random.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Sapien/freq_None/Seed1/Random.cfg--train&
# wait

# KB_GPT
# TAG=PBMC_GPT_GPT4
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_GPT/freq_1+2/Seed1/GRNB2.cfg --train --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_1+2/Seed1/GRNB2.cfg --train --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_3000/KB_GPT/freq_1+2/Seed1/GRNB2.cfg --train --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_4000/KB_GPT/freq_1+2/Seed1/GRNB2.cfg --train --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_GPT/freq_2/Seed1/GRNB2.cfg --train --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_2/Seed1/GRNB2.cfg --train --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_3000/KB_GPT/freq_2/Seed1/GRNB2.cfg --train --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_4000/KB_GPT/freq_2/Seed1/GRNB2.cfg --train --generate&
# wait

# TAG=CTL+KBGPT
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --train&
# wait

# ### CTL
# Stage1
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-PBMC_CTL-1.txt" -e "$LOG_DIR/error-stage1-PBMC_CTL-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/Stage1.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-PBMC_CTL-2.txt" -e "$LOG_DIR/error-stage1-PBMC_CTL-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/Stage1.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-PBMC_CTL-3.txt" -e "$LOG_DIR/error-stage1-PBMC_CTL-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_3000/Stage1.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-PBMC_CTL-4.txt" -e "$LOG_DIR/error-stage1-PBMC_CTL-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_4000/Stage1.cfg --train&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-PBMC_CTL-1.txt" -e "$LOG_DIR/error-stage1-PBMC_CTL-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/Stage1.cfg --generate_cc&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-PBMC_CTL-2.txt" -e "$LOG_DIR/error-stage1-PBMC_CTL-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/Stage1.cfg --generate_cc&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-PBMC_CTL-3.txt" -e "$LOG_DIR/error-stage1-PBMC_CTL-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_3000/Stage1.cfg --generate_cc&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-PBMC_CTL-4.txt" -e "$LOG_DIR/error-stage1-PBMC_CTL-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_4000/Stage1.cfg --generate_cc&
# wait


# ### BoneMarrow
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-BoneMarrow-1.txt" -e "$LOG_DIR/error-stage1-BoneMarrow-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/Stage1.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-BoneMarrow-2.txt" -e "$LOG_DIR/error-stage1-BoneMarrow-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/Stage1.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-BoneMarrow-3.txt" -e "$LOG_DIR/error-stage1-BoneMarrow-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_3000/Stage1.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-BoneMarrow-4.txt" -e "$LOG_DIR/error-stage1-BoneMarrow-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_4000/Stage1.cfg --train&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-BoneMarrow-1.txt" -e "$LOG_DIR/error-stage1-BoneMarrow-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/Stage1.cfg --generate_cc&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-BoneMarrow-2.txt" -e "$LOG_DIR/error-stage1-BoneMarrow-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/Stage1.cfg --generate_cc&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-BoneMarrow-3.txt" -e "$LOG_DIR/error-stage1-BoneMarrow-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_3000/Stage1.cfg --generate_cc&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-BoneMarrow-4.txt" -e "$LOG_DIR/error-stage1-BoneMarrow-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_4000/Stage1.cfg --generate_cc&
# wait

# TAG=BoneMarrow_KB_Llama
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0   python ${PROJECT_DIR}/src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_Llama/freq_2/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1   python ${PROJECT_DIR}/src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Llama/freq_2/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2   python ${PROJECT_DIR}/src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_Sapien/freq_None/Seed1/Llama.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3   python ${PROJECT_DIR}/src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Sapien/freq_None/Seed1/Llama.cfg --train &
# wait

# TAG=PBMMC_CTL_KB_Llama
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0   python ${PROJECT_DIR}/src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/KB_Llama/freq_2/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1   python ${PROJECT_DIR}/src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_Llama/freq_2/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2   python ${PROJECT_DIR}/src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/KB_Sapien/freq_None/Seed1/Llama.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3   python ${PROJECT_DIR}/src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_Sapien/freq_None/Seed1/Llama.cfg --train &
# wait

# Stage1
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-COVID-1.txt" -e "$LOG_DIR/error-stage1-COVID-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/COVID/CrossVal_1000/Stage1.cfg --train&
# wait

TAG=COVID_KB_Sapien
srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0   python ${PROJECT_DIR}/src/main.py --config ${PROJECT_DIR}/configs/COVID/CrossVal_1000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --train&
wait