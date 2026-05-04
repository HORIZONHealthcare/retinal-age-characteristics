#!/bin/bash
# Usage: bash scripts/run.sh --exp_name my_exp --splits_dir /path/to/splits [other overrides...]

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT train.py \
    "$@"
