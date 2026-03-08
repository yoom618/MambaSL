model_name="MambaSingleLayer"
dataset_name="FaceDetection"
tslib_dir="/data/yoom618/TSLib"
gpu_id=0

data_dir="${tslib_dir}/dataset"
checkpoint_dir="${tslib_dir}/checkpoints_best/MambaSL (inceptiontime-setting, 2 more trials)"

python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification_trainlossonly \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "CLS_${dataset_name}" \
  --mamba_projection_type gating \
  --d_model 512 \
  --d_ff 16 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 0 \
  --tv_B 1 \
  --tv_C 1 \
  --use_D 0 \
  --num_kernels 3 \
  --is_training 0 \
  --batch_size 16 \
  --des trainlossonly \
  --itr_start 1 \
  --itr 2 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10