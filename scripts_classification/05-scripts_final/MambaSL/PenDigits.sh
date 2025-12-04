model_name="MambaSingleLayer"
dataset_name="PenDigits"
source_dir="/data/user/MambaSL"
gpu_id=0

data_dir="${source_dir}/dataset"
checkpoint_dir="${source_dir}/checkpoints_best/MambaSL"

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
  --mamba_projection_type gating \
  --d_model 64 \
  --d_ff 1 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 0 \
  --tv_B 1 \
  --tv_C 1 \
  --use_D 0 \
  --num_kernels 3 \
  --is_training 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10