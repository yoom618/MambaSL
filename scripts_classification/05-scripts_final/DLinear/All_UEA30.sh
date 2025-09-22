# the best performing model for all UEA30 dataset
# if there is more than one model, we choose the one with the lowest model size or computation cost
model_name="DLinear"
tslib_dir="/data/yoom618/TSLib"
gpu_id=0

data_dir="${tslib_dir}/dataset"
checkpoint_dir="${tslib_dir}/checkpoints_best/${model_name}"

# ArticularyWordRecognition
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/ArticularyWordRecognition" \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_ArticularyWordRecognition \
  --moving_avg 8 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# AtrialFibrillation
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/AtrialFibrillation" \
  --seq_len 640 \
  --enc_in 2 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_AtrialFibrillation \
  --moving_avg 96 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# BasicMotions
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/BasicMotions" \
  --seq_len 100 \
  --enc_in 6 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_BasicMotions \
  --moving_avg 50 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# CharacterTrajectories
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/CharacterTrajectories" \
  --seq_len 182 \
  --enc_in 3 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_CharacterTrajectories \
  --moving_avg 91 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# Cricket
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/Cricket" \
  --seq_len 1197 \
  --enc_in 6 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_Cricket \
  --moving_avg 240 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# DuckDuckGeese
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/DuckDuckGeese" \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_DuckDuckGeese \
  --moving_avg 122 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# EigenWorms
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/EigenWorms" \
  --seq_len 17984 \
  --enc_in 6 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id "CLS_EigenWorms" \
  --moving_avg 180 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# Epilepsy
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/Epilepsy" \
  --seq_len 206 \
  --enc_in 3 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_Epilepsy \
  --moving_avg 42 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# ERing
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/ERing" \
  --seq_len 65 \
  --enc_in 4 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_ERing \
  --moving_avg 33 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# EthanolConcentration
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/EthanolConcentration" \
  --seq_len 1751 \
  --enc_in 3 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_EthanolConcentration \
  --moving_avg 263 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# FaceDetection
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/FaceDetection" \
  --seq_len 62 \
  --enc_in 144 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_FaceDetection \
  --moving_avg 7 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# FingerMovements
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/FingerMovements" \
  --seq_len 50 \
  --enc_in 28 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_FingerMovements \
  --moving_avg 23 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# HandMovementDirection
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/HandMovementDirection" \
  --seq_len 400 \
  --enc_in 10 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_HandMovementDirection \
  --moving_avg 160 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# Handwriting
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/Handwriting" \
  --seq_len 152 \
  --enc_in 3 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_Handwriting \
  --moving_avg 38 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# Heartbeat
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/Heartbeat" \
  --seq_len 405 \
  --enc_in 61 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_Heartbeat \
  --moving_avg 3 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# InsectWingbeat
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/InsectWingbeat" \
  --seq_len 78 \
  --enc_in 200 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_InsectWingbeat \
  --moving_avg 7 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# JapaneseVowels
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/JapaneseVowels" \
  --seq_len 29 \
  --enc_in 12 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_JapaneseVowels \
  --moving_avg 15 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# Libras
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/Libras" \
  --seq_len 45 \
  --enc_in 2 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_Libras \
  --moving_avg 21 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# LSST
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/LSST" \
  --seq_len 36 \
  --enc_in 6 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_LSST \
  --moving_avg 11 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# MotorImagery
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/MotorImagery" \
  --seq_len 3000 \
  --enc_in 64 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_MotorImagery \
  --moving_avg 1350 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# NATOPS
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/NATOPS" \
  --seq_len 51 \
  --enc_in 24 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_NATOPS \
  --moving_avg 13 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# PEMS-SF
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/PEMS-SF" \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_PEMS-SF \
  --moving_avg 8 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# PenDigits
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/PenDigits" \
  --model ${model_name} \
  --model_id CLS_PenDigits \
  --seq_len 8 \
  --enc_in 2 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --moving_avg 4 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# PhonemeSpectra
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/PhonemeSpectra" \
  --seq_len 217 \
  --enc_in 11 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_PhonemeSpectra \
  --moving_avg 55 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# RacketSports
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/RacketSports" \
  --seq_len 30 \
  --enc_in 6 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_RacketSports \
  --moving_avg 1 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# SelfRegulationSCP1
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/SelfRegulationSCP1" \
  --seq_len 896 \
  --enc_in 6 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_SelfRegulationSCP1 \
  --moving_avg 448 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# SelfRegulationSCP2
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/SelfRegulationSCP2" \
  --seq_len 1152 \
  --enc_in 7 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_SelfRegulationSCP2 \
  --moving_avg 519 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# SpokenArabicDigits
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/SpokenArabicDigits" \
  --seq_len 93 \
  --enc_in 13 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_SpokenArabicDigits \
  --moving_avg 24 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# StandWalkJump
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/StandWalkJump" \
  --seq_len 2500 \
  --enc_in 4 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_StandWalkJump \
  --moving_avg 50 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# UWaveGestureLibrary
python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/UWaveGestureLibrary" \
  --seq_len 315 \
  --enc_in 3 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id CLS_UWaveGestureLibrary \
  --moving_avg 95 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10