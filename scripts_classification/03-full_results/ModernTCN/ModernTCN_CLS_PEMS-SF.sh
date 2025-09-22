# runned in A100 via Google Colab
# commented lines are the ones that cannot be run in A100 Memory

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 1 1 \
  --large_size 13 13 13 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 1 \
  --large_size 13 13 13 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 1 1 \
  --large_size 13 13 13 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 1 1 \
  --large_size 13 13 13 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 1 1 \
  --large_size 13 13 13 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 1 1 \
  --large_size 13 13 13 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 1 1 \
  --large_size 13 13 13 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 1 1 \
  --large_size 9 9 9 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 1 \
  --large_size 9 9 9 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 1 1 \
  --large_size 9 9 9 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 1 1 \
  --large_size 9 9 9 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 1 1 \
  --large_size 9 9 9 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 1 1 \
  --large_size 9 9 9 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 1 1 \
  --large_size 9 9 9 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

## python TSLib/run.py \
##   --use_gpu True \
##   --gpu_type cuda \
##   --gpu 0 \
##   --task_name classification \
##   --data UEA \
##   --root_path ./dataset/PEMS-SF/ \
##   --seq_len 144 \
##   --enc_in 963 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_PEMS-SF \
##   --ffn_ratio 4 \
##   --patch_size 4 \
##   --patch_stride 2 \
##   --num_blocks 1 1 \
##   --large_size 13 13 \
##   --small_size 5 5 \
##   --dims 128 256 \
##   --is_training 1 \
##   --batch_size 1 \
##   --des Exp \
##   --itr 1 \
##   --dropout 0.1 \
##   --learning_rate 0.001 \
##   --train_epochs 50 \
##   --patience 10

## python TSLib/run.py \
##   --use_gpu True \
##   --gpu_type cuda \
##   --gpu 0 \
##   --task_name classification \
##   --data UEA \
##   --root_path ./dataset/PEMS-SF/ \
##   --seq_len 144 \
##   --enc_in 963 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_PEMS-SF \
##   --ffn_ratio 4 \
##   --patch_size 8 \
##   --patch_stride 4 \
##   --num_blocks 1 1 \
##   --large_size 13 13 \
##   --small_size 5 5 \
##   --dims 128 256 \
##   --is_training 1 \
##   --batch_size 1 \
##   --des Exp \
##   --itr 1 \
##   --dropout 0.1 \
##   --learning_rate 0.001 \
##   --train_epochs 50 \
##   --patience 10

## python TSLib/run.py \
##   --use_gpu True \
##   --gpu_type cuda \
##   --gpu 0 \
##   --task_name classification \
##   --data UEA \
##   --root_path ./dataset/PEMS-SF/ \
##   --seq_len 144 \
##   --enc_in 963 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_PEMS-SF \
##   --ffn_ratio 4 \
##   --patch_size 11 \
##   --patch_stride 6 \
##   --num_blocks 1 1 \
##   --large_size 13 13 \
##   --small_size 5 5 \
##   --dims 128 256 \
##   --is_training 1 \
##   --batch_size 1 \
##   --des Exp \
##   --itr 1 \
##   --dropout 0.1 \
##   --learning_rate 0.001 \
##   --train_epochs 50 \
##   --patience 10

## python TSLib/run.py \
##   --use_gpu True \
##   --gpu_type cuda \
##   --gpu 0 \
##   --task_name classification \
##   --data UEA \
##   --root_path ./dataset/PEMS-SF/ \
##   --seq_len 144 \
##   --enc_in 963 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_PEMS-SF \
##   --ffn_ratio 4 \
##   --patch_size 15 \
##   --patch_stride 8 \
##   --num_blocks 1 1 \
##   --large_size 13 13 \
##   --small_size 5 5 \
##   --dims 128 256 \
##   --is_training 1 \
##   --batch_size 1 \
##   --des Exp \
##   --itr 1 \
##   --dropout 0.1 \
##   --learning_rate 0.001 \
##   --train_epochs 50 \
##   --patience 10

## python TSLib/run.py \
##   --use_gpu True \
##   --gpu_type cuda \
##   --gpu 0 \
##   --task_name classification \
##   --data UEA \
##   --root_path ./dataset/PEMS-SF/ \
##   --seq_len 144 \
##   --enc_in 963 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_PEMS-SF \
##   --ffn_ratio 4 \
##   --patch_size 22 \
##   --patch_stride 11 \
##   --num_blocks 1 1 \
##   --large_size 13 13 \
##   --small_size 5 5 \
##   --dims 128 256 \
##   --is_training 1 \
##   --batch_size 1 \
##   --des Exp \
##   --itr 1 \
##   --dropout 0.1 \
##   --learning_rate 0.001 \
##   --train_epochs 50 \
##   --patience 10

## python TSLib/run.py \
##   --use_gpu True \
##   --gpu_type cuda \
##   --gpu 0 \
##   --task_name classification \
##   --data UEA \
##   --root_path ./dataset/PEMS-SF/ \
##   --seq_len 144 \
##   --enc_in 963 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_PEMS-SF \
##   --ffn_ratio 4 \
##   --patch_size 29 \
##   --patch_stride 15 \
##   --num_blocks 1 1 \
##   --large_size 13 13 \
##   --small_size 5 5 \
##   --dims 128 256 \
##   --is_training 1 \
##   --batch_size 1 \
##   --des Exp \
##   --itr 1 \
##   --dropout 0.1 \
##   --learning_rate 0.001 \
##   --train_epochs 50 \
##   --patience 10

## python TSLib/run.py \
##   --use_gpu True \
##   --gpu_type cuda \
##   --gpu 0 \
##   --task_name classification \
##   --data UEA \
##   --root_path ./dataset/PEMS-SF/ \
##   --seq_len 144 \
##   --enc_in 963 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_PEMS-SF \
##   --ffn_ratio 4 \
##   --patch_size 36 \
##   --patch_stride 18 \
##   --num_blocks 1 1 \
##   --large_size 13 13 \
##   --small_size 5 5 \
##   --dims 128 256 \
##   --is_training 1 \
##   --batch_size 1 \
##   --des Exp \
##   --itr 1 \
##   --dropout 0.1 \
##   --learning_rate 0.001 \
##   --train_epochs 50 \
##   --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

## python TSLib/run.py \
##   --use_gpu True \
##   --gpu_type cuda \
##   --gpu 0 \
##   --task_name classification \
##   --data UEA \
##   --root_path ./dataset/PEMS-SF/ \
##   --seq_len 144 \
##   --enc_in 963 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_PEMS-SF \
##   --ffn_ratio 4 \
##   --patch_size 4 \
##   --patch_stride 2 \
##   --num_blocks 1 1 \
##   --large_size 9 9 \
##   --small_size 5 5 \
##   --dims 128 256 \
##   --is_training 1 \
##   --batch_size 1 \
##   --des Exp \
##   --itr 1 \
##   --dropout 0.1 \
##   --learning_rate 0.001 \
##   --train_epochs 50 \
##   --patience 10

## python TSLib/run.py \
##   --use_gpu True \
##   --gpu_type cuda \
##   --gpu 0 \
##   --task_name classification \
##   --data UEA \
##   --root_path ./dataset/PEMS-SF/ \
##   --seq_len 144 \
##   --enc_in 963 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_PEMS-SF \
##   --ffn_ratio 4 \
##   --patch_size 8 \
##   --patch_stride 4 \
##   --num_blocks 1 1 \
##   --large_size 9 9 \
##   --small_size 5 5 \
##   --dims 128 256 \
##   --is_training 1 \
##   --batch_size 1 \
##   --des Exp \
##   --itr 1 \
##   --dropout 0.1 \
##   --learning_rate 0.001 \
##   --train_epochs 50 \
##   --patience 10

## python TSLib/run.py \
##   --use_gpu True \
##   --gpu_type cuda \
##   --gpu 0 \
##   --task_name classification \
##   --data UEA \
##   --root_path ./dataset/PEMS-SF/ \
##   --seq_len 144 \
##   --enc_in 963 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_PEMS-SF \
##   --ffn_ratio 4 \
##   --patch_size 11 \
##   --patch_stride 6 \
##   --num_blocks 1 1 \
##   --large_size 9 9 \
##   --small_size 5 5 \
##   --dims 128 256 \
##   --is_training 1 \
##   --batch_size 1 \
##   --des Exp \
##   --itr 1 \
##   --dropout 0.1 \
##   --learning_rate 0.001 \
##   --train_epochs 50 \
##   --patience 10

## python TSLib/run.py \
##   --use_gpu True \
##   --gpu_type cuda \
##   --gpu 0 \
##   --task_name classification \
##   --data UEA \
##   --root_path ./dataset/PEMS-SF/ \
##   --seq_len 144 \
##   --enc_in 963 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_PEMS-SF \
##   --ffn_ratio 4 \
##   --patch_size 15 \
##   --patch_stride 8 \
##   --num_blocks 1 1 \
##   --large_size 9 9 \
##   --small_size 5 5 \
##   --dims 128 256 \
##   --is_training 1 \
##   --batch_size 1 \
##   --des Exp \
##   --itr 1 \
##   --dropout 0.1 \
##   --learning_rate 0.001 \
##   --train_epochs 50 \
##   --patience 10

## python TSLib/run.py \
##   --use_gpu True \
##   --gpu_type cuda \
##   --gpu 0 \
##   --task_name classification \
##   --data UEA \
##   --root_path ./dataset/PEMS-SF/ \
##   --seq_len 144 \
##   --enc_in 963 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_PEMS-SF \
##   --ffn_ratio 4 \
##   --patch_size 22 \
##   --patch_stride 11 \
##   --num_blocks 1 1 \
##   --large_size 9 9 \
##   --small_size 5 5 \
##   --dims 128 256 \
##   --is_training 1 \
##   --batch_size 1 \
##   --des Exp \
##   --itr 1 \
##   --dropout 0.1 \
##   --learning_rate 0.001 \
##   --train_epochs 50 \
##   --patience 10

## python TSLib/run.py \
##   --use_gpu True \
##   --gpu_type cuda \
##   --gpu 0 \
##   --task_name classification \
##   --data UEA \
##   --root_path ./dataset/PEMS-SF/ \
##   --seq_len 144 \
##   --enc_in 963 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_PEMS-SF \
##   --ffn_ratio 4 \
##   --patch_size 29 \
##   --patch_stride 15 \
##   --num_blocks 1 1 \
##   --large_size 9 9 \
##   --small_size 5 5 \
##   --dims 128 256 \
##   --is_training 1 \
##   --batch_size 1 \
##   --des Exp \
##   --itr 1 \
##   --dropout 0.1 \
##   --learning_rate 0.001 \
##   --train_epochs 50 \
##   --patience 10

## python TSLib/run.py \
##   --use_gpu True \
##   --gpu_type cuda \
##   --gpu 0 \
##   --task_name classification \
##   --data UEA \
##   --root_path ./dataset/PEMS-SF/ \
##   --seq_len 144 \
##   --enc_in 963 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_PEMS-SF \
##   --ffn_ratio 4 \
##   --patch_size 36 \
##   --patch_stride 18 \
##   --num_blocks 1 1 \
##   --large_size 9 9 \
##   --small_size 5 5 \
##   --dims 128 256 \
##   --is_training 1 \
##   --batch_size 1 \
##   --des Exp \
##   --itr 1 \
##   --dropout 0.1 \
##   --learning_rate 0.001 \
##   --train_epochs 50 \
##   --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 8 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

## python TSLib/run.py \
##   --use_gpu True \
##   --gpu_type cuda \
##   --gpu 0 \
##   --task_name classification \
##   --data UEA \
##   --root_path ./dataset/PEMS-SF/ \
##   --seq_len 144 \
##   --enc_in 963 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_PEMS-SF \
##   --ffn_ratio 4 \
##   --patch_size 4 \
##   --patch_stride 2 \
##   --num_blocks 1 \
##   --large_size 13 \
##   --small_size 5 \
##   --dims 256 \
##   --is_training 1 \
##   --batch_size 1 \
##   --des Exp \
##   --itr 1 \
##   --dropout 0.1 \
##   --learning_rate 0.001 \
##   --train_epochs 50 \
##   --patience 10

## python TSLib/run.py \
##   --use_gpu True \
##   --gpu_type cuda \
##   --gpu 0 \
##   --task_name classification \
##   --data UEA \
##   --root_path ./dataset/PEMS-SF/ \
##   --seq_len 144 \
##   --enc_in 963 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_PEMS-SF \
##   --ffn_ratio 4 \
##   --patch_size 8 \
##   --patch_stride 4 \
##   --num_blocks 1 \
##   --large_size 13 \
##   --small_size 5 \
##   --dims 256 \
##   --is_training 1 \
##   --batch_size 1 \
##   --des Exp \
##   --itr 1 \
##   --dropout 0.1 \
##   --learning_rate 0.001 \
##   --train_epochs 50 \
##   --patience 10

## python TSLib/run.py \
##   --use_gpu True \
##   --gpu_type cuda \
##   --gpu 0 \
##   --task_name classification \
##   --data UEA \
##   --root_path ./dataset/PEMS-SF/ \
##   --seq_len 144 \
##   --enc_in 963 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_PEMS-SF \
##   --ffn_ratio 4 \
##   --patch_size 11 \
##   --patch_stride 6 \
##   --num_blocks 1 \
##   --large_size 13 \
##   --small_size 5 \
##   --dims 256 \
##   --is_training 1 \
##   --batch_size 1 \
##   --des Exp \
##   --itr 1 \
##   --dropout 0.1 \
##   --learning_rate 0.001 \
##   --train_epochs 50 \
##   --patience 10

## python TSLib/run.py \
##   --use_gpu True \
##   --gpu_type cuda \
##   --gpu 0 \
##   --task_name classification \
##   --data UEA \
##   --root_path ./dataset/PEMS-SF/ \
##   --seq_len 144 \
##   --enc_in 963 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_PEMS-SF \
##   --ffn_ratio 4 \
##   --patch_size 15 \
##   --patch_stride 8 \
##   --num_blocks 1 \
##   --large_size 13 \
##   --small_size 5 \
##   --dims 256 \
##   --is_training 1 \
##   --batch_size 1 \
##   --des Exp \
##   --itr 1 \
##   --dropout 0.1 \
##   --learning_rate 0.001 \
##   --train_epochs 50 \
##   --patience 10

## python TSLib/run.py \
##   --use_gpu True \
##   --gpu_type cuda \
##   --gpu 0 \
##   --task_name classification \
##   --data UEA \
##   --root_path ./dataset/PEMS-SF/ \
##   --seq_len 144 \
##   --enc_in 963 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_PEMS-SF \
##   --ffn_ratio 4 \
##   --patch_size 22 \
##   --patch_stride 11 \
##   --num_blocks 1 \
##   --large_size 13 \
##   --small_size 5 \
##   --dims 256 \
##   --is_training 1 \
##   --batch_size 1 \
##   --des Exp \
##   --itr 1 \
##   --dropout 0.1 \
##   --learning_rate 0.001 \
##   --train_epochs 50 \
##   --patience 10

## python TSLib/run.py \
##   --use_gpu True \
##   --gpu_type cuda \
##   --gpu 0 \
##   --task_name classification \
##   --data UEA \
##   --root_path ./dataset/PEMS-SF/ \
##   --seq_len 144 \
##   --enc_in 963 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_PEMS-SF \
##   --ffn_ratio 4 \
##   --patch_size 29 \
##   --patch_stride 15 \
##   --num_blocks 1 \
##   --large_size 13 \
##   --small_size 5 \
##   --dims 256 \
##   --is_training 1 \
##   --batch_size 1 \
##   --des Exp \
##   --itr 1 \
##   --dropout 0.1 \
##   --learning_rate 0.001 \
##   --train_epochs 50 \
##   --patience 10

## python TSLib/run.py \
##   --use_gpu True \
##   --gpu_type cuda \
##   --gpu 0 \
##   --task_name classification \
##   --data UEA \
##   --root_path ./dataset/PEMS-SF/ \
##   --seq_len 144 \
##   --enc_in 963 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_PEMS-SF \
##   --ffn_ratio 4 \
##   --patch_size 36 \
##   --patch_stride 18 \
##   --num_blocks 1 \
##   --large_size 13 \
##   --small_size 5 \
##   --dims 256 \
##   --is_training 1 \
##   --batch_size 1 \
##   --des Exp \
##   --itr 1 \
##   --dropout 0.1 \
##   --learning_rate 0.001 \
##   --train_epochs 50 \
##   --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 32 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 32 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 32 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 32 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 32 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 32 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 4 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 32 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 1 1 \
  --large_size 13 13 13 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 1 \
  --large_size 13 13 13 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 1 1 \
  --large_size 13 13 13 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 1 1 \
  --large_size 13 13 13 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 1 1 \
  --large_size 13 13 13 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 1 1 \
  --large_size 13 13 13 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 1 1 \
  --large_size 13 13 13 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 1 1 \
  --large_size 9 9 9 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 1 \
  --large_size 9 9 9 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 1 1 \
  --large_size 9 9 9 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 1 1 \
  --large_size 9 9 9 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 1 1 \
  --large_size 9 9 9 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 1 1 \
  --large_size 9 9 9 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 1 1 \
  --large_size 9 9 9 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 8 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 8 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 256 \
  --is_training 1 \
  --batch_size 8 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 32 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 32 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 32 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 32 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 32 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 32 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 2 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 32 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 1 1 \
  --large_size 13 13 13 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 1 \
  --large_size 13 13 13 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 1 1 \
  --large_size 13 13 13 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 1 1 \
  --large_size 13 13 13 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 1 1 \
  --large_size 13 13 13 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 1 1 \
  --large_size 13 13 13 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 1 1 \
  --large_size 13 13 13 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 1 1 \
  --large_size 9 9 9 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 1 \
  --large_size 9 9 9 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 1 1 \
  --large_size 9 9 9 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 1 1 \
  --large_size 9 9 9 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 1 1 \
  --large_size 9 9 9 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 1 1 \
  --large_size 9 9 9 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 1 1 \
  --large_size 9 9 9 \
  --small_size 5 5 5 \
  --dims 32 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 1 \
  --large_size 13 13 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 128 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 64 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 1 \
  --large_size 9 9 \
  --small_size 5 5 \
  --dims 32 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 256 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 64 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 4 \
  --patch_stride 2 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 32 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 32 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 11 \
  --patch_stride 6 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 32 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 15 \
  --patch_stride 8 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 32 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \a
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 22 \
  --patch_stride 11 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 32 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 29 \
  --patch_stride 15 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 32 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

python TSLib/run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  --task_name classification \
  --data UEA \
  --root_path ./dataset/PEMS-SF/ \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_PEMS-SF \
  --ffn_ratio 1 \
  --patch_size 36 \
  --patch_stride 18 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 32 \
  --is_training 1 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10

