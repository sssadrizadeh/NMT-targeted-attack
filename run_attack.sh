#!/bin/bash
#SBATCH --job-name=test_fr_marian_TransFool(n=2)(10-0-16)(first-id-bestword)
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-10:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10240
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/sasa/icassp2023

# Verify working directory
echo $(pwd)

# Print gpu configuration for this job
nvidia-smi

# Verify gpu allocation (should be 1 GPU)
echo $CUDA_VISIBLE_DEVICES

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate NMT_pre
# conda activate black_attack



# python TransFool/white_box_attack.py --num_samples 1000 --w_sim 10 --w_perp 0 --lr 0.020 --model_name marian --target_lang fr --dataset_config_name fr-en --dataset_name wmt14 --result_folder abl --attack_target guerre --ref_or_first first 
python TransFool/white_box_attack.py --num_samples 1000 --w_sim 10 --w_perp 0 --lr 0.016 --model_name marian --target_lang fr --dataset_config_name fr-en --dataset_name wmt14 --result_folder abl --Nth 2 --ref_or_first first 
# python TransFool/white_box_attack.py --num_samples 100 --w_sim 10 4 2 --w_perp 0 --lr 0.020 --model_name mbart --target_lang fr --dataset_config_name fr-en --dataset_name wmt14 --result_folder flexpos --attack_target guerre --ref_or_first first 
# python TransFool/white_box_attack.py --num_samples 100 --w_sim 10 4 2 --w_perp 0 --lr 0.020 --model_name mbart --target_lang fr --dataset_config_name fr-en --dataset_name wmt14 --result_folder flexpos --Nth 2 --ref_or_first first 
# python Seq2Sick/white_box_attack.py --num_samples 1000 --model_name marian --target_lang de --dataset_config_name de-en --dataset_name wmt14 --result_folder final --lr 8 --gr True --gl True --const 0.001 --attack_target krieg --ref_or_first first
# python Seq2Sick/white_box_attack.py --num_samples 1000 --model_name marian --target_lang de --dataset_config_name de-en --dataset_name wmt14 --result_folder final --lr 8 --gr True --gl True --const 0.001 --Nth 100 --ref_or_first first
# python Seq2Sick/white_box_attack.py --num_samples 1000 --model_name mbart --target_lang de --dataset_config_name de-en --dataset_name wmt14 --result_folder results --lr 8 --gr True --gl True --const 0.001 --attack_target krieg  #--Nth 2
# python Seq2Sick/white_box_attack.py --num_samples 1000 --model_name mbart --target_lang fr --dataset_config_name fr-en --dataset_name wmt14 --result_folder results --lr 160 --gr True --gl True --const 0.00005 --Nth 2


conda deactivate