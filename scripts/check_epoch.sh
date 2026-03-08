#!/bin/bash

TAG="newdst"
BASE_DIR="outputs"

# 调整表头，只留两列
printf "%-40s | %-15s\n" "Job Name" "Current Epoch"
echo "----------------------------------------------------------"

# 遍历符合 Tag 的文件夹
for job_dir in "$BASE_DIR"/"$TAG"_*; do
    if [ -d "$job_dir" ]; then
        job_name=$(basename "$job_dir")
        log_file="$job_dir/log.txt"

        # 计算逻辑
        # wc -l < file 效率最高，且不输出文件名
        k=$(wc -l < "$log_file")
        current_epoch=$((k))

        # 输出结果
        printf "%-40s | %-15s\n" "$job_name" "$current_epoch"
    fi
done