model_name="TimesNet"
dataset_name="EthanolConcentration"
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
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 3 \
  --top_k 3 \
  --num_kernels 6 \
  --d_model 8 \
  --d_ff 16 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10

python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 3 \
  --top_k 2 \
  --num_kernels 6 \
  --d_model 8 \
  --d_ff 16 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10

python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 3 \
  --top_k 1 \
  --num_kernels 6 \
  --d_model 8 \
  --d_ff 16 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10