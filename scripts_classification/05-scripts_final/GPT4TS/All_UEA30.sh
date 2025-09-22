# the best performing model for all UEA30 dataset
# if there is more than one model, we choose the one with the lowest model size or computation cost
model_name="GPT4TS"
tslib_dir="/data/yoom618/TSLib"
gpu_id=0

data_dir="${tslib_dir}/dataset"
checkpoint_dir="${tslib_dir}/checkpoints_best/${model_name}"
huggingface_cache_dir="${tslib_dir}/huggingface"

# ArticularyWordRecognition
dataset_name="ArticularyWordRecognition"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id CLS_${dataset_name} \
  --e_layers 3 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 6 \
  --patch_stride 3 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# AtrialFibrillation
dataset_name="AtrialFibrillation"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 3 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 27 \
  --patch_stride 27 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# BasicMotions
dataset_name="BasicMotions"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 4 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 20 \
  --patch_stride 5 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# CharacterTrajectories
dataset_name="CharacterTrajectories"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 6 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 14 \
  --patch_stride 14 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# Cricket
dataset_name="Cricket"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 3 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 67 \
  --patch_stride 67 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# DuckDuckGeese
dataset_name="DuckDuckGeese"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 4 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 15 \
  --patch_stride 8 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# EigenWorms
dataset_name="EigenWorms"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 6 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 720 \
  --patch_stride 360 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# Epilepsy
dataset_name="Epilepsy"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 4 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 16 \
  --patch_stride 4 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# ERing
dataset_name="ERing"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 6 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 10 \
  --patch_stride 10 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# EthanolConcentration
dataset_name="EthanolConcentration"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 3 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 80 \
  --patch_stride 40 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# FaceDetection
dataset_name="FaceDetection"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 6 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 3 \
  --patch_stride 1 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# FingerMovements
dataset_name="FingerMovements"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 3 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 8 \
  --patch_stride 8 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# HandMovementDirection
dataset_name="HandMovementDirection"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 5 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 80 \
  --patch_stride 80 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# Handwriting
dataset_name="Handwriting"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 6 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 8 \
  --patch_stride 8 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# Heartbeat
dataset_name="Heartbeat"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 3 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 17 \
  --patch_stride 17 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# InsectWingbeat
dataset_name="InsectWingbeat"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 6 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 2 \
  --patch_stride 1 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# JapaneseVowels
dataset_name="JapaneseVowels"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 3 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 3 \
  --patch_stride 2 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# Libras
dataset_name="Libras"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 3 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 8 \
  --patch_stride 8 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# LSST
dataset_name="LSST"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 3 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 2 \
  --patch_stride 1 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# MotorImagery
dataset_name="MotorImagery"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 3 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 150 \
  --patch_stride 38 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# NATOPS
dataset_name="NATOPS"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 3 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 5 \
  --patch_stride 5 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# PEMS-SF
dataset_name="PEMS-SF"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 5 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 6 \
  --patch_stride 6 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# PenDigits
dataset_name="PenDigits"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 5 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 2 \
  --patch_stride 2 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# PhonemeSpectra
dataset_name="PhonemeSpectra"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 5 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 14 \
  --patch_stride 7 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# RacketSports
dataset_name="RacketSports"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 5 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 4 \
  --patch_stride 1 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# SelfRegulationSCP1
dataset_name="SelfRegulationSCP1"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 3 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 112 \
  --patch_stride 56 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# SelfRegulationSCP2
dataset_name="SelfRegulationSCP2"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 3 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 116 \
  --patch_stride 29 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# SpokenArabicDigits
dataset_name="SpokenArabicDigits"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 3 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 6 \
  --patch_stride 2 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# StandWalkJump
dataset_name="StandWalkJump"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 3 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 132 \
  --patch_stride 33 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

# UWaveGestureLibrary
dataset_name="UWaveGestureLibrary"
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --huggingface_cache_dir ${huggingface_cache_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 3 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 17 \
  --patch_stride 17 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10