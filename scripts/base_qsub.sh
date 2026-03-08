# ------ basic settings
#$ -S /bin/bash
#$ -j y
#$ -cwd

# ------ memory
#$ -l tmem=40G

# ------ runtime
#$ -l h_rt=168:00:00

# ------ gpu
#$ -l gpu=true,gpu_type=a6000
#$ -pe gpu 1

#$ -o /SAN/ioo/Alzeye10/yiqunlin/retinal_age_prediction/_qsub_log_files/$JOB_NAME.o$JOB_ID

# -----------------------------
hostname
date
nvidia-smi

conda activate torch-1.13

mkdir -p outputs/$JOB_NAME

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune.py \
    --savemodel \
    --batch_size 32 \
    --world_size 1 \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 \
    --input_size 224 \
    --modalities $MODALITIES \
    --train_mode finetune \
    --clip_grad 1.0 \
    --mm_model \
    --patient_condition $PACOND \
    --task $JOB_NAME > outputs/$JOB_NAME/$JOB_NAME-$JOB_ID.log

date