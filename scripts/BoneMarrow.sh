# Stage1
srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-BoneMarrow-1.txt" -e "$LOG_DIR/error-stage1-BoneMarrow-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/Stage1.cfg --train&
srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-BoneMarrow-2.txt" -e "$LOG_DIR/error-stage1-BoneMarrow-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/Stage1.cfg --train&
srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-BoneMarrow-3.txt" -e "$LOG_DIR/error-stage1-BoneMarrow-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_3000/Stage1.cfg --train&
srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-BoneMarrow-4.txt" -e "$LOG_DIR/error-stage1-BoneMarrow-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_4000/Stage1.cfg --train&
wait

srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-BoneMarrow-1.txt" -e "$LOG_DIR/error-stage1-BoneMarrow-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/Stage1.cfg --generate_cc&
srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-BoneMarrow-2.txt" -e "$LOG_DIR/error-stage1-BoneMarrow-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/Stage1.cfg --generate_cc&
srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-BoneMarrow-3.txt" -e "$LOG_DIR/error-stage1-BoneMarrow-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_3000/Stage1.cfg --generate_cc&
srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage1-BoneMarrow-4.txt" -e "$LOG_DIR/error-stage1-BoneMarrow-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_4000/Stage1.cfg --generate_cc&
wait

# Stage2
# KB_Sapien
# TAG=BoneMarrow_Sapien_GRNB2
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_3000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_4000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --train&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_3000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_4000/KB_Sapien/freq_None/Seed1/GRNB2.cfg --generate&
# wait

# TAG=BoneMarrow_Sapien_Random
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_Sapien/freq_None/Seed1/Random.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Sapien/freq_None/Seed1/Random.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_3000/KB_Sapien/freq_None/Seed1/Random.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_4000/KB_Sapien/freq_None/Seed1/Random.cfg --train&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_Sapien/freq_None/Seed1/Random.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Sapien/freq_None/Seed1/Random.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_3000/KB_Sapien/freq_None/Seed1/Random.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_4000/KB_Sapien/freq_None/Seed1/Random.cfg --generate&
# wait

# TAG=BoneMarrow_Sapien_GPT4
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_Sapien/freq_None/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Sapien/freq_None/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_3000/KB_Sapien/freq_None/Seed1/GPT4.cfg --train&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_4000/KB_Sapien/freq_None/Seed1/GPT4.cfg --train&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_Sapien/freq_None/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_Sapien/freq_None/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_3000/KB_Sapien/freq_None/Seed1/GPT4.cfg --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_4000/KB_Sapien/freq_None/Seed1/GPT4.cfg --generate&
# wait

# KB_GPT
# TAG=BoneMarrow_GPT_GPT4
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_GPT/freq_1+2/Seed1/GRNB2.cfg --train --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_GPT/freq_1+2/Seed1/GRNB2.cfg --train --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_3000/KB_GPT/freq_1+2/Seed1/GRNB2.cfg --train --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_4000/KB_GPT/freq_1+2/Seed1/GRNB2.cfg --train --generate&
# wait

# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-1.txt" -e "$LOG_DIR/error-stage2-${TAG}-1.txt"  env CUDA_VISIBLE_DEVICES=0  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_1000/KB_GPT/freq_2/Seed1/GRNB2.cfg --train --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-2.txt" -e "$LOG_DIR/error-stage2-${TAG}-2.txt"  env CUDA_VISIBLE_DEVICES=1  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_2000/KB_GPT/freq_2/Seed1/GRNB2.cfg --train --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-3.txt" -e "$LOG_DIR/error-stage2-${TAG}-3.txt"  env CUDA_VISIBLE_DEVICES=2  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_3000/KB_GPT/freq_2/Seed1/GRNB2.cfg --train --generate&
# srun --exclusive --gres=gpu:1 --ntasks=1 --nodes=1 --ntasks-per-node=1 -o "$LOG_DIR/out-stage2-${TAG}-4.txt" -e "$LOG_DIR/error-stage2-${TAG}-4.txt"  env CUDA_VISIBLE_DEVICES=3  python src/main.py --config ${PROJECT_DIR}/configs/BoneMarrow/CrossVal_4000/KB_GPT/freq_2/Seed1/GRNB2.cfg --train --generate&
# wait