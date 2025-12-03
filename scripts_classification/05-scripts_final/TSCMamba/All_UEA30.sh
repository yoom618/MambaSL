# the best performing model for all UEA30 dataset
# if there is more than one model, we choose the one with the lowest model size or computation cost
model_name="TSCMamba"
tslib_dir="/data/yoom618/TSLib"
gpu_id=0

data_dir="${tslib_dir}/dataset"
checkpoint_dir="${tslib_dir}/checkpoints_best/${model_name}"

# ArticularyWordRecognition
dataset_name="ArticularyWordRecognition"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 128 \
  --e_layers 2 \
  --expand 2 \
  --d_conv 4 \
  --d_ff 5 \
  --half_rocket 0 \
  --additive_fusion 0 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 1 \
  --channel_token_mixing 0 \
  --num_kernels 3 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# AtrialFibrillation
dataset_name="AtrialFibrillation"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 256 \
  --e_layers 1 \
  --expand 2 \
  --d_conv 4 \
  --d_ff 2 \
  --half_rocket 0 \
  --additive_fusion 0 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 0 \
  --channel_token_mixing 0 \
  --num_kernels 13 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# BasicMotions
dataset_name="BasicMotions"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 64 \
  --e_layers 1 \
  --expand 1 \
  --d_conv 4 \
  --d_ff 1 \
  --half_rocket 1 \
  --additive_fusion 0 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 1 \
  --channel_token_mixing 0 \
  --num_kernels 3 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# CharacterTrajectories
dataset_name="CharacterTrajectories"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 256 \
  --e_layers 1 \
  --expand 1 \
  --d_conv 4 \
  --d_ff 5 \
  --half_rocket 1 \
  --additive_fusion 1 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 1 \
  --channel_token_mixing 0 \
  --num_kernels 4 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# Cricket
dataset_name="Cricket"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 64 \
  --e_layers 1 \
  --expand 1 \
  --d_conv 4 \
  --d_ff 4 \
  --half_rocket 0 \
  --additive_fusion 0 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 1 \
  --channel_token_mixing 0 \
  --num_kernels 24 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# DuckDuckGeese
dataset_name="DuckDuckGeese"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 64 \
  --e_layers 2 \
  --expand 1 \
  --d_conv 4 \
  --d_ff 2 \
  --half_rocket 0 \
  --additive_fusion 0 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 0 \
  --channel_token_mixing 0 \
  --num_kernels 6 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# EigenWorms
dataset_name="EigenWorms"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 256 \
  --e_layers 1 \
  --expand 1 \
  --d_conv 4 \
  --d_ff 3 \
  --half_rocket 0 \
  --additive_fusion 0 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 0 \
  --channel_token_mixing 0 \
  --num_kernels 360 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# Epilepsy
dataset_name="Epilepsy"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 128 \
  --e_layers 1 \
  --expand 1 \
  --d_conv 4 \
  --d_ff 4 \
  --half_rocket 1 \
  --additive_fusion 0 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 1 \
  --channel_token_mixing 0 \
  --num_kernels 5 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# ERing
dataset_name="ERing"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 128 \
  --e_layers 2 \
  --expand 2 \
  --d_conv 4 \
  --d_ff 4 \
  --half_rocket 1 \
  --additive_fusion 0 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 1 \
  --channel_token_mixing 0 \
  --num_kernels 3 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# EthanolConcentration
dataset_name="EthanolConcentration"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 64 \
  --e_layers 1 \
  --expand 1 \
  --d_conv 4 \
  --d_ff 2 \
  --half_rocket 0 \
  --additive_fusion 0 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 0 \
  --channel_token_mixing 0 \
  --num_kernels 36 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# FaceDetection
dataset_name="FaceDetection"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 128 \
  --e_layers 1 \
  --expand 2 \
  --d_conv 4 \
  --d_ff 2 \
  --half_rocket 1 \
  --additive_fusion 1 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 0 \
  --channel_token_mixing 0 \
  --num_kernels 3 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# FingerMovements
dataset_name="FingerMovements"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 64 \
  --e_layers 1 \
  --expand 1 \
  --d_conv 4 \
  --d_ff 3 \
  --half_rocket 1 \
  --additive_fusion 1 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 0 \
  --channel_token_mixing 0 \
  --num_kernels 3 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# HandMovementDirection
dataset_name="HandMovementDirection"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 64 \
  --e_layers 1 \
  --expand 2 \
  --d_conv 4 \
  --d_ff 2 \
  --half_rocket 1 \
  --additive_fusion 1 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 1 \
  --channel_token_mixing 0 \
  --num_kernels 8 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# Handwriting
dataset_name="Handwriting"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 256 \
  --e_layers 2 \
  --expand 1 \
  --d_conv 4 \
  --d_ff 6 \
  --half_rocket 0 \
  --additive_fusion 1 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 1 \
  --channel_token_mixing 0 \
  --num_kernels 4 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# Heartbeat
dataset_name="Heartbeat"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 64 \
  --e_layers 1 \
  --expand 2 \
  --d_conv 4 \
  --d_ff 1 \
  --half_rocket 1 \
  --additive_fusion 0 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 0 \
  --channel_token_mixing 0 \
  --num_kernels 9 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# InsectWingbeat
dataset_name="InsectWingbeat"
# python run.py \
#   --use_gpu True \
#   --gpu_type cuda \
#   --gpu ${gpu_id} \
#   --task_name classification \
#   --data UEA \
#   --root_path "${data_dir}/${dataset_name}" \
#   --checkpoints ${checkpoint_dir} \
#   --model ${model_name} \
#   --model_id "CLS_${dataset_name}" \
#   --d_model 256 \
#   --e_layers 2 \
#   --expand 2 \
#   --d_conv 4 \
#   --d_ff 5 \
#   --half_rocket 1 \
#   --additive_fusion 1 \
#   --only_forward_scan 0 \
#   --flip_dir 2 \
#   --reverse_flip 0 \
#   --patch_size 8 \
#   --rescale_size 64 \
#   --variation 64 \
#   --initial_focus 1.0 \
#   --no_rocket 0 \
#   --max_pooling 1 \
#   --channel_token_mixing 0 \
#   --num_kernels 3 \
#   --is_training 0 \
#   --batch_size 16 \
#   --des Exp \
#   --itr 1 \
#   --dropout 0.1 \
#   --learning_rate 0.001 \
#   --train_epochs 100 \
#   --patience 10

# JapaneseVowels
dataset_name="JapaneseVowels"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 256 \
  --e_layers 2 \
  --expand 1 \
  --d_conv 4 \
  --d_ff 3 \
  --half_rocket 1 \
  --additive_fusion 0 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 1 \
  --channel_token_mixing 0 \
  --num_kernels 3 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# Libras
dataset_name="Libras"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 256 \
  --e_layers 1 \
  --expand 2 \
  --d_conv 4 \
  --d_ff 3 \
  --half_rocket 0 \
  --additive_fusion 1 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 1 \
  --channel_token_mixing 0 \
  --num_kernels 3 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# LSST
dataset_name="LSST"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 256 \
  --e_layers 2 \
  --expand 2 \
  --d_conv 4 \
  --d_ff 9 \
  --half_rocket 0 \
  --additive_fusion 1 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 0 \
  --channel_token_mixing 0 \
  --num_kernels 3 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# MotorImagery
dataset_name="MotorImagery"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 128 \
  --e_layers 1 \
  --expand 1 \
  --d_conv 4 \
  --d_ff 2 \
  --half_rocket 1 \
  --additive_fusion 1 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 0 \
  --channel_token_mixing 0 \
  --num_kernels 60 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# NATOPS
dataset_name="NATOPS"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 64 \
  --e_layers 2 \
  --expand 2 \
  --d_conv 4 \
  --d_ff 4 \
  --half_rocket 1 \
  --additive_fusion 1 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 1 \
  --channel_token_mixing 0 \
  --num_kernels 3 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# PEMS-SF
dataset_name="PEMS-SF"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 128 \
  --e_layers 2 \
  --expand 1 \
  --d_conv 4 \
  --d_ff 4 \
  --half_rocket 1 \
  --additive_fusion 1 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 0 \
  --channel_token_mixing 0 \
  --num_kernels 3 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10
# PenDigits
dataset_name="PenDigits"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 128 \
  --e_layers 1 \
  --expand 1 \
  --d_conv 4 \
  --d_ff 4 \
  --half_rocket 0 \
  --additive_fusion 1 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 0 \
  --channel_token_mixing 0 \
  --num_kernels 3 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# PhonemeSpectra
dataset_name="PhonemeSpectra"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 256 \
  --e_layers 1 \
  --expand 2 \
  --d_conv 4 \
  --d_ff 7 \
  --half_rocket 0 \
  --additive_fusion 0 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 0 \
  --channel_token_mixing 0 \
  --num_kernels 5 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# RacketSports
dataset_name="RacketSports"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 256 \
  --e_layers 1 \
  --expand 2 \
  --d_conv 4 \
  --d_ff 3 \
  --half_rocket 0 \
  --additive_fusion 1 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 1 \
  --channel_token_mixing 0 \
  --num_kernels 3 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# SelfRegulationSCP1
dataset_name="SelfRegulationSCP1"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 64 \
  --e_layers 1 \
  --expand 1 \
  --d_conv 4 \
  --d_ff 3 \
  --half_rocket 1 \
  --additive_fusion 0 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 1 \
  --channel_token_mixing 0 \
  --num_kernels 18 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# SelfRegulationSCP2
dataset_name="SelfRegulationSCP2"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 128 \
  --e_layers 1 \
  --expand 1 \
  --d_conv 4 \
  --d_ff 1 \
  --half_rocket 1 \
  --additive_fusion 1 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 0 \
  --channel_token_mixing 0 \
  --num_kernels 24 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# SpokenArabicDigits
dataset_name="SpokenArabicDigits"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 128 \
  --e_layers 1 \
  --expand 1 \
  --d_conv 4 \
  --d_ff 4 \
  --half_rocket 1 \
  --additive_fusion 1 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 1 \
  --channel_token_mixing 0 \
  --num_kernels 3 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# StandWalkJump
dataset_name="StandWalkJump"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 128 \
  --e_layers 1 \
  --expand 1 \
  --d_conv 4 \
  --d_ff 1 \
  --half_rocket 0 \
  --additive_fusion 1 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 0 \
  --channel_token_mixing 0 \
  --num_kernels 50 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# UWaveGestureLibrary
dataset_name="UWaveGestureLibrary"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 256 \
  --e_layers 1 \
  --expand 1 \
  --d_conv 4 \
  --d_ff 4 \
  --half_rocket 0 \
  --additive_fusion 1 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 1 \
  --channel_token_mixing 0 \
  --num_kernels 7 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10