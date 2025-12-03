model_name="MambaML"
dataset_name="CharacterTrajectories"
tslib_dir="/data/yoom618/TSLib"
gpu_id=0

data_dir="${tslib_dir}/dataset"
checkpoint_dir="${tslib_dir}/checkpoints_best/MambaSL (multilayer)"

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
  --e_layers 3 \
  --mamba_projection_type gating \
  --d_model 64 \
  --d_ff 2 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 1 \
  --tv_B 0 \
  --tv_C 1 \
  --use_D 0 \
  --num_kernels 4 \
  --is_training 0 \
  --batch_size 16 \
  --des gating4multilayer \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10