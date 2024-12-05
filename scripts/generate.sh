#!/bin/bash

#SBATCH --job-name=groundgan-generate
#SBATCH --time=2:00:00
#SBATCH --account hai_steerllms
#SBATCH --partition develbooster
#SBATCH --mail-user=tejumade.afonja@cispa.de
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mem 32G
#SBATCH --cpus-per-task=4
#SBATCH -o output-%x.txt 
#SBATCH -e error-%x.txt

nvidia-smi
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


PROJECT_DIR=/p/project1/hai_steerllms/GRouNdGAN
LOG_DIR=${PROJECT_DIR}/scripts/logs
mkdir ${LOG_DIR}

source ${PROJECT_DIR}/activate.sh
cd ${PROJECT_DIR}
# python src/main.py --config /p/project1/hai_diffprivtab/teju/GRouNdGAN/configs/causal_gan.cfg --preprocess --create_grn
# python src/main.py --config /p/project1/hai_diffprivtab/teju/GRouNdGAN/configs/causal_gan.cfg --create_grn
# tensorboard --logdir="/p/project1/hai_diffprivtab/teju/GRouNdGAN/results/GRouNdGAN/TensorBoard" --host 0.0.0.0 --load_fast false &
# srun --exclusive python src/main.py --config /p/project1/hai_diffprivtab/teju/GRouNdGAN/configs/causal_gan.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-g1.txt" -e "$LOG_DIR/error-g1.txt"  env CUDA_VISIBLE_DEVICES=0 python src/main.py --config /p/project1/hai_diffprivtab/teju/GRouNdGAN/configs/PBMC/causal_gan_GT.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-g2.txt" -e "$LOG_DIR/error-g2.txt"  env CUDA_VISIBLE_DEVICES=1 python src/main.py --config /p/project1/hai_diffprivtab/teju/GRouNdGAN/configs/PBMC/causal_gan_GPT.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-g3.txt" -e "$LOG_DIR/error-g3.txt"  env CUDA_VISIBLE_DEVICES=2 python src/main.py --config /p/project1/hai_diffprivtab/teju/GRouNdGAN/configs/PBMC/causal_gan_Random.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-gen.txt" -e "$LOG_DIR/error-stage1-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/Stage1.cfg --generate_cc&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-gen.txt" -e "$LOG_DIR/error-stage1-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/Stage1.cfg --generate_cc&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-gen.txt" -e "$LOG_DIR/error-stage1-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/Stage1.cfg --generate_cc&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-gen.txt" -e "$LOG_DIR/error-stage1-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/Stage1.cfg --generate_cc&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-PBMC-gen.txt" -e "$LOG_DIR/error-PBMC-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config /p/project1/hai_diffprivtab/teju/GRouNdGAN/configs/PBMC/causal_gan_GT.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-BM-gen.txt" -e "$LOG_DIR/error-BM-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config /p/project1/hai_diffprivtab/teju/GRouNdGAN/configs/BoneMarrow/causal_gan_GT.cfg --generate&
# # srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-CTL-gen.txt" -e "$LOG_DIR/error-CTL-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config /p/project1/hai_diffprivtab/teju/GRouNdGAN/configs/PBMC_CTL/causal_gan_GT.cfg --generate&
# wait


# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-tf10-CTL-gen.txt" -e "$LOG_DIR/error-tf10-CTL-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config /p/project1/hai_diffprivtab/teju/GRouNdGAN/configs/PBMC_CTL/causal_gan_GT.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-tf15-CTL-gen.txt" -e "$LOG_DIR/error-tf15-CTL-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config /p/project1/hai_diffprivtab/teju/GRouNdGAN/configs/PBMC_CTL/EXP_TF_k/causal_gan_TF15.cfg --generate&
# wait


# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-BM-gen.txt" -e "$LOG_DIR/error-BM-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config /p/project1/hai_diffprivtab/teju/GRouNdGAN/configs/BoneMarrow/causal_gan_GPT.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-BM-gen.txt" -e "$LOG_DIR/error-BM-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config /p/project1/hai_diffprivtab/teju/GRouNdGAN/configs/BoneMarrow/causal_gan_Random.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-PBMC-gen.txt" -e "$LOG_DIR/error-PBMC-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config /p/project1/hai_diffprivtab/teju/GRouNdGAN/configs/PBMC/causal_gan_Random.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-CTL-gen.txt" -e "$LOG_DIR/error-CTL-gen.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config /p/project1/hai_diffprivtab/teju/GRouNdGAN/configs/PBMC_CTL/causal_gan_Random.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-LU-gen.txt" -e "$LOG_DIR/error-LU-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config /p/project1/hai_diffprivtab/teju/GRouNdGAN/configs/LinearUniform/causal_gan_GT.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-LU-gen.txt" -e "$LOG_DIR/error-LU-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config /p/project1/hai_diffprivtab/teju/GRouNdGAN/configs/LinearUniform/causal_gan_Stage1.cfg --generate_cc&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-PBMC-gen.txt" -e "$LOG_DIR/error-PBMC-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config /p/project1/hai_diffprivtab/teju/GRouNdGAN/configs/PBMC/causal_gan_GPT.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-CTL-gen.txt" -e "$LOG_DIR/error-CTL-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config /p/project1/hai_diffprivtab/teju/GRouNdGAN/configs/PBMC_CTL/causal_gan_GPT.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-PBMC-gen.txt" -e "$LOG_DIR/error-PBMC-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config /p/project1/hai_diffprivtab/teju/GRouNdGAN/configs/PBMC/causal_gan_Random.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-CTL-gen.txt" -e "$LOG_DIR/error-CTL-gen.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config /p/project1/hai_diffprivtab/teju/GRouNdGAN/configs/PBMC_CTL/causal_gan_Random.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Sapien/freq_None/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Sapien/freq_None/Seed1/Random.cfg --generate&
# # # srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-gen.txt" -e "$LOG_DIR/error-stage1-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/Stage1.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_1/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_1/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_2/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-gen.txt" -e "$LOG_DIR/error-stage1-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_2/Seed1/GRNB2.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-LU-gen.txt" -e "$LOG_DIR/error-LU-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/Stage1.cfg --generate_cc&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-LU-gen.txt" -e "$LOG_DIR/error-LU-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/Stage1.cfg --generate_cc&
# # srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-LU-gen.txt" -e "$LOG_DIR/error-LU-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/Stage1.cfg --generate_cc&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-LU-gen.txt" -e "$LOG_DIR/error-LU-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/CrossVal_1000/Stage1.cfg --generate_cc&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-LU-gen.txt" -e "$LOG_DIR/error-LU-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/CrossVal_2000/Stage1.cfg --generate_cc&
# # srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-LU-gen.txt" -e "$LOG_DIR/error-LU-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/Stage1.cfg --generate_cc&
# wait


# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-LU-gen.txt" -e "$LOG_DIR/error-LU-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/Stage1.cfg --generate_cc&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-LU-gen.txt" -e "$LOG_DIR/error-LU-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/Stage1.cfg --generate_cc&
# # srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-LU-gen.txt" -e "$LOG_DIR/error-LU-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/Stage1.cfg --generate_cc&
# wait


# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-LU-gen.txt" -e "$LOG_DIR/error-LU-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/Stage1.cfg --generate_cc&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-LU-gen.txt" -e "$LOG_DIR/error-LU-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/Stage1.cfg --generate_cc&
# # srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-LU-gen.txt" -e "$LOG_DIR/error-LU-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/Stage1.cfg --generate_cc&
# wait


# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-LU-gen.txt" -e "$LOG_DIR/error-LU-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/CrossVal_2000/Stage1.cfg --generate_cc&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-LU-gen.txt" -e "$LOG_DIR/error-LU-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/Stage1.cfg --generate_cc&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-LU-gen.txt" -e "$LOG_DIR/error-LU-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/Stage1.cfg --generate_cc&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_GPT/freq_1/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_GPT/freq_1/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_1/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_1/Seed1/GRNB2.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_GPT/freq_2/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_GPT/freq_2/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_2/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_2/Seed1/GRNB2.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Sapien/freq_None/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Sapien/freq_None/Seed1/Random.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Sapien/freq_None/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Sapien/freq_None/Seed1/Random.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/KB_Sapien/freq_None/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/KB_Sapien/freq_None/Seed1/Random.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_Sapien/freq_None/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_Sapien/freq_None/Seed1/Random.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_Sapien/freq_None/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_Sapien/freq_None/Seed1/Random.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Sapien/freq_None/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Sapien/freq_None/Seed1/Random.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/CrossVal_1000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/CrossVal_2000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --generate&
# wait



# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_Sapien/freq_None/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_Sapien/freq_None/Seed1/Random.cfg --generate&
# # srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-gen.txt" -e "$LOG_DIR/error-stage1-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/KB_Sapien/freq_2/Seed1/GRNB2.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_Sapien/freq_None/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_Sapien/freq_None/Seed1/Random.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-gen.txt" -e "$LOG_DIR/error-stage1-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/CrossVal_1000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Sapien/freq_None/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Sapien/freq_None/Seed1/Random.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-gen.txt" -e "$LOG_DIR/error-stage1-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/LinearUniform/CrossVal_2000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --generate&
# wait
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --generate&
# # srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/KB_Sapien/freq_None/Seed1/Random.cfg --generate&
# # srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --generate&
# # srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_Sapien/freq_None/Seed1/Random.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_2/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_2/Seed1/GRNB2.cfg --generate&
# wait

# 
# Synthetic Loop
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-SL-LU-gen.txt" -e "$LOG_DIR/error-SL-LU-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config /p/project1/hai_diffprivtab/teju/GRouNdGAN/configs/LinearUniform/synthetic_loop/stage1/iter0.cfg --generate_cc&
# wait
# configs/BoneMarrow/CrossVal_1000/KB_GPT/freq_1/Seed1/GPT4.cfg
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_GPT/freq_1/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_GPT/freq_1/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_GPT/freq_1/Seed1/Random.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_GPT/freq_2/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_GPT/freq_2/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_GPT/freq_2/Seed1/Random.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_1/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_1/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_1/Seed1/Random.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_2/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_2/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_2/Seed1/Random.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Llama/freq_1/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Llama/freq_1/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Llama/freq_1/Seed1/Random.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Llama/freq_2/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Llama/freq_2/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Llama/freq_2/Seed1/Random.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/freq_1/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/freq_1/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/freq_1/Seed1/Random.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_2/Seed1/GRNB2-tmp.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/freq_2/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/freq_2/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/freq_2/Seed1/Random.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_GPT/freq_2/Seed1/GRNB2.cfg --generate&
# wait


# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Random/freq_None/Seed1/Random.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Random/freq_None/Seed1/Random.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_Random/freq_None/Seed1/Random.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Random/freq_None/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/KB_Random/freq_None/Seed1/Random.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_Random/freq_None/Seed1/Random.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Sapien/freq_None/Seed1/Llama.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Sapien/freq_None/Seed1/Llama.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_Random/freq_None/Seed1/Random.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Random/freq_None/Seed1/Random.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Random/freq_None/Seed1/Random.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Random/freq_None/Seed1/Random.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/KB_Random/freq_None/Seed1/Random.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_Random/freq_None/Seed1/Random.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Llama/no_kb_gpt/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/no_kb_gpt/Seed1/GRNB2.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_1000/KB_Llama/no_kb_human/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/no_kb_human/Seed1/GRNB2.cfg --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/freq_1/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/freq_1/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC/CrossVal_2000/KB_Llama/freq_1/Seed1/Random.cfg --generate&
# wait


srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_Llama/freq_2/Seed1/GRNB2.cfg --generate&
srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Llama/freq_2/Seed1/GRNB2.cfg --generate&
srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_Sapien/freq_2/Seed1/Llama.cfg --generate&
srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Sapien/freq_2/Seed1/Llama.cfg --generate&
wait

srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/KB_Llama/freq_2/Seed1/GRNB2.cfg --generate&
srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_Llama/freq_2/Seed1/GRNB2.cfg --generate&
srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_1000/KB_Sapien/freq_2/Seed1/Llama.cfg --generate&
srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-gen.txt" -e "$LOG_DIR/error-stage2-gen.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/PBMC_CTL/CrossVal_2000/KB_Sapien/freq_2/Seed1/Llama.cfg --generate&
wait