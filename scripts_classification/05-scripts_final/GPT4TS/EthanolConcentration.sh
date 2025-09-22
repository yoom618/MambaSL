model_name="GPT4TS"
dataset_name="EthanolConcentration"
tslib_dir="/data/yoom618/TSLib"
gpu_id=0

data_dir="${tslib_dir}/dataset"
checkpoint_dir="${tslib_dir}/checkpoints_best/${model_name}"
huggingface_cache_dir="${tslib_dir}/huggingface"

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