# model=MambaSL_CLS
# dataset=ADFTD
# exp=additional
# echo "Running ./scripts_classification/scripts_mamba/${exp}/${model}_${dataset}_repeat.sh"
# echo "Result will be saved in ./scripts_classification/results/${model}_${dataset}_repeat.out"
# nohup bash ./scripts_classification/scripts_mamba/${exp}/${model}_${dataset}_repeat.sh > ./scripts_classification/results/${model}_${dataset}_repeat.out &


python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 3 \
  --task_name classification_medical \
  --data ADFTD \
  --root_path /data/yoom618/TSLib/dataset/ADFTD \
  --seq_len 256 \
  --enc_in 19 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --batch_size 128 \
  --checkpoints /data/yoom618/TSLib/checkpoints \
  --model MambaSingleLayer \
  --model_id CLS_ADFTD \
  --mamba_projection_type gating \
  --d_model 1024 \
  --d_ff 16 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 0 \
  --tv_B 1 \
  --tv_C 0 \
  --use_D 0 \
  --num_kernels 6 \
  --is_training 1 \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  --swa True \
  --itr 4 \
  --itr_start 1


python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 3 \
  --task_name classification_medical \
  --data ADFTD \
  --root_path /data/yoom618/TSLib/dataset/ADFTD \
  --seq_len 256 \
  --enc_in 19 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --batch_size 128 \
  --checkpoints /data/yoom618/TSLib/checkpoints \
  --model MambaSingleLayer \
  --model_id CLS_ADFTD \
  --mamba_projection_type gating \
  --d_model 1024 \
  --d_ff 8 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 1 \
  --tv_B 1 \
  --tv_C 1 \
  --use_D 0 \
  --num_kernels 6 \
  --is_training 1 \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  --swa True \
  --itr 4 \
  --itr_start 1


python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 3 \
  --task_name classification_medical \
  --data ADFTD \
  --root_path /data/yoom618/TSLib/dataset/ADFTD \
  --seq_len 256 \
  --enc_in 19 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --batch_size 128 \
  --checkpoints /data/yoom618/TSLib/checkpoints \
  --model MambaSingleLayer \
  --model_id CLS_ADFTD \
  --mamba_projection_type gating \
  --d_model 1024 \
  --d_ff 2 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 0 \
  --tv_B 0 \
  --tv_C 1 \
  --use_D 0 \
  --num_kernels 6 \
  --is_training 1 \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  --swa True \
  --itr 4 \
  --itr_start 1


python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 3 \
  --task_name classification_medical \
  --data ADFTD \
  --root_path /data/yoom618/TSLib/dataset/ADFTD \
  --seq_len 256 \
  --enc_in 19 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --batch_size 128 \
  --checkpoints /data/yoom618/TSLib/checkpoints \
  --model MambaSingleLayer \
  --model_id CLS_ADFTD \
  --mamba_projection_type gating \
  --d_model 128 \
  --d_ff 4 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 0 \
  --tv_B 0 \
  --tv_C 0 \
  --use_D 0 \
  --num_kernels 6 \
  --is_training 1 \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  --swa True \
  --itr 4 \
  --itr_start 1


python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 3 \
  --task_name classification_medical \
  --data ADFTD \
  --root_path /data/yoom618/TSLib/dataset/ADFTD \
  --seq_len 256 \
  --enc_in 19 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --batch_size 128 \
  --checkpoints /data/yoom618/TSLib/checkpoints \
  --model MambaSingleLayer \
  --model_id CLS_ADFTD \
  --mamba_projection_type gating \
  --d_model 32 \
  --d_ff 8 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 0 \
  --tv_B 0 \
  --tv_C 0 \
  --use_D 0 \
  --num_kernels 6 \
  --is_training 1 \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  --swa True \
  --itr 4 \
  --itr_start 1


python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 3 \
  --task_name classification_medical \
  --data ADFTD \
  --root_path /data/yoom618/TSLib/dataset/ADFTD \
  --seq_len 256 \
  --enc_in 19 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --batch_size 128 \
  --checkpoints /data/yoom618/TSLib/checkpoints \
  --model MambaSingleLayer \
  --model_id CLS_ADFTD \
  --mamba_projection_type gating \
  --d_model 32 \
  --d_ff 1 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 0 \
  --tv_B 1 \
  --tv_C 1 \
  --use_D 0 \
  --num_kernels 6 \
  --is_training 1 \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  --swa True \
  --itr 4 \
  --itr_start 1


