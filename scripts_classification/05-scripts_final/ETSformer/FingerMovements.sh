model_name="ETSformer"
dataset_name="FingerMovements"
tslib_dir="/data/yoom618/TSLib"
gpu_id=0

data_dir="${tslib_dir}/dataset"
checkpoint_dir="${tslib_dir}/checkpoints_best/${model_name}"

python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints ${checkpoint_dir} \
  --c_out 0 \
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --e_layers 2 \
  --n_heads 8 \
  --d_model 256 \
  --d_ff 2048 \
  --top_k 2 \
  --d_layers 2 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10