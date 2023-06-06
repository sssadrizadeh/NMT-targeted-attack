#!/bin/bash
#SBATCH --job-name=eval_TransFool_marian_en_fr(n=2)(10,4,2-0-20)(0)(first-ids-bestword-flexpos)
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-10:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10240
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/sahar.sadrizadeh/Targeted_Intern

# Verify working directory
echo $(pwd)

# Print gpu configuration for this job
nvidia-smi

# Verify gpu allocation (should be 1 GPU)
echo $CUDA_VISIBLE_DEVICES

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate NMT


# python EvaluateAttack.py --target_lang de2 --attack_alg WSLS --result_folder results --target_model_name mbart
# python EvaluateAttack.py --num_samples 1000 --target_model_name marian --target_lang fr --result_folder fixpos --attack_type white_box --attack_alg TransFool --w_sim 10 4 2 --w_perp 0 --lr 0.02 --bad_sim 0 --attack_target guerre
python EvaluateAttack.py --num_samples 1000 --target_model_name marian --target_lang fr --result_folder fixpos --attack_type white_box --attack_alg TransFool --w_sim 10 4 2 --w_perp 0 --lr 0.02 --bad_sim 0 --Nth 2
# python EvaluateAttack.py --num_samples 100 --target_model_name marian --target_lang de --result_folder flexpos --attack_type white_box --attack_alg TransFool  --w_sim 10 4 2 --w_perp 0 --lr 0.02  --bad_sim 0.7 --attack_target krieg
# python EvaluateAttack.py --num_samples 1000 --target_model_name marian --target_lang fr --result_folder final --attack_type white_box --attack_alg Seq2Sick --lr 8 --const 0.001 --bad_sim 0 --Nth 10
# python EvaluateAttack.py --num_samples 1000 --target_model_name marian --target_lang fr --result_folder final --attack_type white_box --attack_alg Seq2Sick --lr 8 --const 0.001 --bad_sim 0 --Nth 2
# python EvaluateAttack.py --num_samples 100 --target_model_name mbart --target_lang de --result_folder flexpos --attack_type white_box --attack_alg Seq2Sick --lr 160 --const 0.00005 --bad_sim 0 --attack_target krieg
# python EvaluateAttack.py --num_samples 100 --target_model_name mbart --target_lang fr --result_folder flexpos --attack_type white_box --attack_alg Seq2Sick  --lr 160 --const 0.00005 --bad_sim 0.7 --Nth 2
# python EvaluateAttack.py --num_samples 1000 --target_model_name marian --target_lang fr --result_folder results --attack_type LM --attack_alg kNN --max-swap 4

# python EvaluateAttack.py --num_samples 3000 --target_model_name mbart --target_lang fr --result_folder results --attack_type two_lang --attack_alg TransFool --bleu 0.4 --w_sim 20 --w_perp 1.8 --lr 0.016
# python EvaluateAttack.py --num_samples 2000 --target_model_name mbart --target_lang zh --result_folder results --attack_type black_box --attack_alg TransFool --bleu 0.4 --w_sim 20 --w_perp 1.8 --lr 0.016
# python EvaluateAttack.py --num_samples 2000 --target_model_name mbart --target_lang zh --result_folder results --attack_type black_box --attack_alg kNN --max-swap 4
# python EvaluateAttack.py --num_samples 2000 --target_model_name mbart --target_lang zh --result_folder results --attack_type black_box --attack_alg Seq2Sick  

# python EvaluateAttack.py --num_samples 2000 --target_model_name google --target_lang zh --result_folder results --attack_type google --attack_alg TransFool

# python TER.py --name en_zh --target_model_name mbart --target_lang zh --attack_alg WSLS --result_folder results --attack_type black_box

conda deactivate