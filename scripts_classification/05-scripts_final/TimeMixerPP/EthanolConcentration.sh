model_name="TimeMixerPP"
dataset_name="EthanolConcentration"
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
  --model ${model_name} \
  --model_id "CLS_${dataset_name}" \
  --down_sampling_method conv \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --num_kernels 6 \
  --e_layers 4 \
  --d_model 32 \
  --d_ff 64 \
  --top_k 3 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10