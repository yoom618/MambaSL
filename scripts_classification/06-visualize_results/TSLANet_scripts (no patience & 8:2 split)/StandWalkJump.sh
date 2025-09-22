model_name="TSLANet"
dataset_name="StandWalkJump"
tslib_dir="/data/yoom618/TSLib"
gpu_id=0

data_dir="${tslib_dir}/dataset"
checkpoint_dir="${tslib_dir}/checkpoints_best/${model_name}"

# below all have the same performance

python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "23_55_27" \
    --model_id "CLS_${dataset_name}" \
    --depth 3 \
    --emb_dim 32 \
    --mlp_ratio 1.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 188

python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "00_55_48" \
    --model_id "CLS_${dataset_name}" \
    --depth 2 \
    --emb_dim 64 \
    --mlp_ratio 1.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 188

python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "00_58_14" \
    --model_id "CLS_${dataset_name}" \
    --depth 2 \
    --emb_dim 64 \
    --mlp_ratio 1.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 500

python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "00_24_04" \
    --model_id "CLS_${dataset_name}" \
    --depth 2 \
    --emb_dim 128 \
    --mlp_ratio 3.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 188

python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "01_23_32" \
    --model_id "CLS_${dataset_name}" \
    --depth 1 \
    --emb_dim 256 \
    --mlp_ratio 2.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 125

python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "01_24_21" \
    --model_id "CLS_${dataset_name}" \
    --depth 1 \
    --emb_dim 256 \
    --mlp_ratio 2.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 188

python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "01_26_04" \
    --model_id "CLS_${dataset_name}" \
    --depth 1 \
    --emb_dim 256 \
    --mlp_ratio 2.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 375

python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "01_17_35" \
    --model_id "CLS_${dataset_name}" \
    --depth 1 \
    --emb_dim 256 \
    --mlp_ratio 3.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 125
