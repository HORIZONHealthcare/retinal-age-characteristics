TASK=alzeye+pacond_CFP_healthy

python main_finetune.py \
    --eval \
    --input_size 224 \
    --modalities CFP \
    --mm_model \
    --output_dir ./outputs/ \
    --task $TASK \
    --resume ./outputs/$TASK/checkpoint.pth \
    --dataset_name ukb