# runned in A100 via Google Colab
# commented lines are the ones that cannot be run in A100 Memory

## python TSLib/run.py \
##   --use_gpu True \
##   --gpu_type cuda \
##   --gpu 0 \
##   --task_name classification \
##   --data UEA \
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 7 \
##   --patch_stride 4 \
##   --num_blocks 1 1 1 \
##   --large_size 13 13 13 \
##   --small_size 5 5 5 \
##   --dims 32 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 14 \
##   --patch_stride 7 \
##   --num_blocks 1 1 1 \
##   --large_size 13 13 13 \
##   --small_size 5 5 5 \
##   --dims 32 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 21 \
##   --patch_stride 11 \
##   --num_blocks 1 1 1 \
##   --large_size 13 13 13 \
##   --small_size 5 5 5 \
##   --dims 32 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 27 \
##   --patch_stride 14 \
##   --num_blocks 1 1 1 \
##   --large_size 13 13 13 \
##   --small_size 5 5 5 \
##   --dims 32 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 41 \
##   --patch_stride 21 \
##   --num_blocks 1 1 1 \
##   --large_size 13 13 13 \
##   --small_size 5 5 5 \
##   --dims 32 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 54 \
##   --patch_stride 27 \
##   --num_blocks 1 1 1 \
##   --large_size 13 13 13 \
##   --small_size 5 5 5 \
##   --dims 32 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 68 \
##   --patch_stride 34 \
##   --num_blocks 1 1 1 \
##   --large_size 13 13 13 \
##   --small_size 5 5 5 \
##   --dims 32 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 7 \
##   --patch_stride 4 \
##   --num_blocks 1 1 1 \
##   --large_size 9 9 9 \
##   --small_size 5 5 5 \
##   --dims 32 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 14 \
##   --patch_stride 7 \
##   --num_blocks 1 1 1 \
##   --large_size 9 9 9 \
##   --small_size 5 5 5 \
##   --dims 32 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 21 \
##   --patch_stride 11 \
##   --num_blocks 1 1 1 \
##   --large_size 9 9 9 \
##   --small_size 5 5 5 \
##   --dims 32 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 27 \
##   --patch_stride 14 \
##   --num_blocks 1 1 1 \
##   --large_size 9 9 9 \
##   --small_size 5 5 5 \
##   --dims 32 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 41 \
##   --patch_stride 21 \
##   --num_blocks 1 1 1 \
##   --large_size 9 9 9 \
##   --small_size 5 5 5 \
##   --dims 32 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 54 \
##   --patch_stride 27 \
##   --num_blocks 1 1 1 \
##   --large_size 9 9 9 \
##   --small_size 5 5 5 \
##   --dims 32 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 68 \
##   --patch_stride 34 \
##   --num_blocks 1 1 1 \
##   --large_size 9 9 9 \
##   --small_size 5 5 5 \
##   --dims 32 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 7 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 14 \
##   --patch_stride 7 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 21 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 27 \
##   --patch_stride 14 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 41 \
##   --patch_stride 21 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 54 \
##   --patch_stride 27 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 68 \
##   --patch_stride 34 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 7 \
##   --patch_stride 4 \
##   --num_blocks 1 1 \
##   --large_size 13 13 \
##   --small_size 5 5 \
##   --dims 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 14 \
##   --patch_stride 7 \
##   --num_blocks 1 1 \
##   --large_size 13 13 \
##   --small_size 5 5 \
##   --dims 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 21 \
##   --patch_stride 11 \
##   --num_blocks 1 1 \
##   --large_size 13 13 \
##   --small_size 5 5 \
##   --dims 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 27 \
##   --patch_stride 14 \
##   --num_blocks 1 1 \
##   --large_size 13 13 \
##   --small_size 5 5 \
##   --dims 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 41 \
##   --patch_stride 21 \
##   --num_blocks 1 1 \
##   --large_size 13 13 \
##   --small_size 5 5 \
##   --dims 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 54 \
##   --patch_stride 27 \
##   --num_blocks 1 1 \
##   --large_size 13 13 \
##   --small_size 5 5 \
##   --dims 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 68 \
##   --patch_stride 34 \
##   --num_blocks 1 1 \
##   --large_size 13 13 \
##   --small_size 5 5 \
##   --dims 64 128 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 68 \
  --patch_stride 34 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 7 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 14 \
##   --patch_stride 7 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 21 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 27 \
##   --patch_stride 14 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 41 \
##   --patch_stride 21 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 54 \
##   --patch_stride 27 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 68 \
##   --patch_stride 34 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 7 \
##   --patch_stride 4 \
##   --num_blocks 1 1 \
##   --large_size 9 9 \
##   --small_size 5 5 \
##   --dims 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 14 \
##   --patch_stride 7 \
##   --num_blocks 1 1 \
##   --large_size 9 9 \
##   --small_size 5 5 \
##   --dims 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 21 \
##   --patch_stride 11 \
##   --num_blocks 1 1 \
##   --large_size 9 9 \
##   --small_size 5 5 \
##   --dims 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 27 \
##   --patch_stride 14 \
##   --num_blocks 1 1 \
##   --large_size 9 9 \
##   --small_size 5 5 \
##   --dims 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 41 \
##   --patch_stride 21 \
##   --num_blocks 1 1 \
##   --large_size 9 9 \
##   --small_size 5 5 \
##   --dims 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 54 \
##   --patch_stride 27 \
##   --num_blocks 1 1 \
##   --large_size 9 9 \
##   --small_size 5 5 \
##   --dims 64 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 68 \
##   --patch_stride 34 \
##   --num_blocks 1 1 \
##   --large_size 9 9 \
##   --small_size 5 5 \
##   --dims 64 128 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 68 \
  --patch_stride 34 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 7 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 14 \
##   --patch_stride 7 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 21 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 27 \
##   --patch_stride 14 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 41 \
##   --patch_stride 21 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 54 \
##   --patch_stride 27 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 68 \
##   --patch_stride 34 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 7 \
##   --patch_stride 4 \
##   --num_blocks 1 \
##   --large_size 13 \
##   --small_size 5 \
##   --dims 128 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 4 \
##   --patch_size 14 \
##   --patch_stride 7 \
##   --num_blocks 1 \
##   --large_size 13 \
##   --small_size 5 \
##   --dims 128 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 21 \
  --patch_stride 11 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 1 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 27 \
  --patch_stride 14 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 2 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 41 \
  --patch_stride 21 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 54 \
  --patch_stride 27 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 2 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 68 \
  --patch_stride 34 \
  --num_blocks 1 \
  --large_size 13 \
  --small_size 5 \
  --dims 128 \
  --is_training 1 \
  --batch_size 2 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 68 \
  --patch_stride 34 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 4 \
  --patch_size 68 \
  --patch_stride 34 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 68 \
  --patch_stride 34 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 68 \
  --patch_stride 34 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 2 \
##   --patch_size 7 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 2 \
##   --patch_size 14 \
##   --patch_stride 7 \
##   --num_blocks 1 1 \
##   --large_size 13 13 \
##   --small_size 5 5 \
##   --dims 128 256 \
##   --is_training 1 \
##   --batch_size 4 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 2 \
##   --patch_size 21 \
##   --patch_stride 11 \
##   --num_blocks 1 1 \
##   --large_size 13 13 \
##   --small_size 5 5 \
##   --dims 128 256 \
##   --is_training 1 \
##   --batch_size 4 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 2 \
##   --patch_size 27 \
##   --patch_stride 14 \
##   --num_blocks 1 1 \
##   --large_size 13 13 \
##   --small_size 5 5 \
##   --dims 128 256 \
##   --is_training 1 \
##   --batch_size 4 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 2 \
##   --patch_size 41 \
##   --patch_stride 21 \
##   --num_blocks 1 1 \
##   --large_size 13 13 \
##   --small_size 5 5 \
##   --dims 128 256 \
##   --is_training 1 \
##   --batch_size 4 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 2 \
##   --patch_size 54 \
##   --patch_stride 27 \
##   --num_blocks 1 1 \
##   --large_size 13 13 \
##   --small_size 5 5 \
##   --dims 128 256 \
##   --is_training 1 \
##   --batch_size 4 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 2 \
##   --patch_size 68 \
##   --patch_stride 34 \
##   --num_blocks 1 1 \
##   --large_size 13 13 \
##   --small_size 5 5 \
##   --dims 128 256 \
##   --is_training 1 \
##   --batch_size 4 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 68 \
  --patch_stride 34 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 68 \
  --patch_stride 34 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 2 \
##   --patch_size 7 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 2 \
##   --patch_size 14 \
##   --patch_stride 7 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 2 \
##   --patch_size 21 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 2 \
##   --patch_size 27 \
##   --patch_stride 14 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 2 \
##   --patch_size 41 \
##   --patch_stride 21 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 2 \
##   --patch_size 54 \
##   --patch_stride 27 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 2 \
##   --patch_size 68 \
##   --patch_stride 34 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 68 \
  --patch_stride 34 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 68 \
  --patch_stride 34 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 2 \
##   --patch_size 7 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 2 \
##   --patch_size 14 \
##   --patch_stride 7 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 2 \
##   --patch_size 21 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 2 \
##   --patch_size 27 \
##   --patch_stride 14 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 2 \
##   --patch_size 41 \
##   --patch_stride 21 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 2 \
##   --patch_size 54 \
##   --patch_stride 27 \
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
##   --root_path ./dataset/DuckDuckGeese \
##   --seq_len 270 \
##   --enc_in 1345 \
##   --label_len 0 \
##   --pred_len 0 \
##   --c_out 0 \
##   --checkpoints ./checkpoints \
##   --model ModernTCN \
##   --model_id CLS_DuckDuckGeese \
##   --ffn_ratio 2 \
##   --patch_size 68 \
##   --patch_stride 34 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 68 \
  --patch_stride 34 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 68 \
  --patch_stride 34 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 2 \
  --patch_size 68 \
  --patch_stride 34 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 68 \
  --patch_stride 34 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 68 \
  --patch_stride 34 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 7 \
  --patch_stride 4 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 68 \
  --patch_stride 34 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 68 \
  --patch_stride 34 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 68 \
  --patch_stride 34 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 7 \
  --patch_stride 4 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 68 \
  --patch_stride 34 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 68 \
  --patch_stride 34 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 68 \
  --patch_stride 34 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 68 \
  --patch_stride 34 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 68 \
  --patch_stride 34 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 68 \
  --patch_stride 34 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 14 \
  --patch_stride 7 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 27 \
  --patch_stride 14 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 41 \
  --patch_stride 21 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 54 \
  --patch_stride 27 \
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
  --root_path ./dataset/DuckDuckGeese \
  --seq_len 270 \
  --enc_in 1345 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ./checkpoints \
  --model ModernTCN \
  --model_id CLS_DuckDuckGeese \
  --ffn_ratio 1 \
  --patch_size 68 \
  --patch_stride 34 \
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

