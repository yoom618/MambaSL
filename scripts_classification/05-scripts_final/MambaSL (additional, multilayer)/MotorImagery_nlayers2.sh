model_name="MambaMultiLayer"
dataset_name="MotorImagery"
resource_dir="/data/yoom618/TSLib"
gpu_id=0

data_dir="${resource_dir}/dataset"
checkpoint_dir="${resource_dir}/checkpoints_best/MambaSL (multilayer)"

python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --e_layers 2 \
  --mamba_projection_type gating \
  --d_model 256 \
  --d_ff 1 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 1 \
  --tv_B 1 \
  --tv_C 1 \
  --use_D 0 \
  --num_kernels 60 \
  --is_training 0 \
  --batch_size 16 \
  --des gating4multilayer \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10