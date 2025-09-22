model_name="FEDformer"
dataset_name="StandWalkJump"
tslib_dir="/data/yoom618/TSLib"
gpu_id=0

data_dir="${tslib_dir}/dataset"
checkpoint_dir="${tslib_dir}/checkpoints_best/${model_name}"

# below all have the same performance

python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 2 \
  --n_heads 8 \
  --d_model 32 \
  --d_ff 256 \
  --moving_avg 500 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 2 \
  --n_heads 8 \
  --d_model 32 \
  --d_ff 512 \
  --moving_avg 250 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 2 \
  --n_heads 8 \
  --d_model 128 \
  --d_ff 128 \
  --moving_avg 13 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 2 \
  --n_heads 8 \
  --d_model 256 \
  --d_ff 256 \
  --moving_avg 1000 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 2 \
  --n_heads 8 \
  --d_model 512 \
  --d_ff 512 \
  --moving_avg 1250 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10