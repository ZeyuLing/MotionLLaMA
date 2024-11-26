#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

TOTAL_GPUS=$(($GPUS * $NNODES))

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch ${@:4}
