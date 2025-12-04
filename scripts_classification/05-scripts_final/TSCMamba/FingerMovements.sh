model_name="TSCMamba"
dataset_name="FingerMovements"
source_dir="/data/user/MambaSL"
gpu_id=0

data_dir="${source_dir}/dataset"
checkpoint_dir="${source_dir}/checkpoints_best/${model_name}"

python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA4TSCMamba \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --d_model 64 \
  --e_layers 1 \
  --expand 1 \
  --d_conv 4 \
  --d_ff 3 \
  --half_rocket 1 \
  --additive_fusion 1 \
  --only_forward_scan 0 \
  --flip_dir 2 \
  --reverse_flip 0 \
  --patch_size 8 \
  --rescale_size 64 \
  --variation 64 \
  --initial_focus 1.0 \
  --no_rocket 0 \
  --max_pooling 0 \
  --channel_token_mixing 0 \
  --num_kernels 3 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10