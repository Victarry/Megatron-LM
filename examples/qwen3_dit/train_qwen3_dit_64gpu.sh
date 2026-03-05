#!/bin/bash
# Qwen3-DiT (94L MoE, MHA, no_mask) — 64 GPU (8 nodes) Benchmark Scripts
#
# Model: 94 layers, h=4096, 64 heads (MHA), 128 experts, topk=8, SwiGLU
# Data: mock data, seq_length configured per experiment
# Cluster: SLURM with enroot containers
#
# Usage:
#   # Choose experiment via EXPERIMENT env var:
#   EXPERIMENT=fsdp_mbs1_moe_recomp    bash train_qwen3_dit_64gpu.sh   # FSDP MBS=1 moe recomp (527 TFLOP/s)
#   EXPERIMENT=fsdp_mbs1_moeact_recomp bash train_qwen3_dit_64gpu.sh   # FSDP MBS=1 moe_act recomp (560 TFLOP/s)
#   EXPERIMENT=fsdp_mbs2_moe_recomp    bash train_qwen3_dit_64gpu.sh   # FSDP MBS=2 moe recomp (640 TFLOP/s)
#   EXPERIMENT=pp8_vpp4_mbs1           bash train_qwen3_dit_64gpu.sh   # PP8/VPP4 MBS=1 (645 TFLOP/s)
#   EXPERIMENT=fsdp_cp4_seq26k         bash train_qwen3_dit_64gpu.sh   # FSDP CP=4 seq=26464 (645 TFLOP/s)
#   EXPERIMENT=pp8_vpp4_cp4_seq26k     bash train_qwen3_dit_64gpu.sh   # PP8/VPP4/CP4 seq=26464 (676 TFLOP/s)
#
# All TFLOP/s numbers are corrected for no_mask full attention.
# Tested on BIA cluster (B300 GPUs, 268 GB HBM3e).

set -euo pipefail

# ============================================================================
# User-configurable
# ============================================================================
EXPERIMENT="${EXPERIMENT:-fsdp_mbs1_moe_recomp}"
NNODES="${NNODES:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/coreai_devtech_all/denliu/sqsh/megatron-moe_pytorch25.12-deepep.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre/:/lustre/}"
ACCOUNT="${ACCOUNT:-coreai_devtech_all}"
PARTITION="${PARTITION:-batch}"
TIME_LIMIT="${TIME_LIMIT:-4:00:00}"
MEGATRON_PATH="${MEGATRON_PATH:-$(cd "$(dirname "$0")/../.." && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-/lustre/fsw/coreai_devtech_all/${USER}/benchmarking/qwen3-dit}"

# ============================================================================
# Model architecture (shared across all experiments)
# ============================================================================
MODEL_ARGS=(
    --num-layers 94
    --hidden-size 4096
    --ffn-hidden-size 12288
    --num-attention-heads 64
    --kv-channels 128
    --max-position-embeddings 32768
    --qk-layernorm
    --attention-mask-type no_mask
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --swiglu
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --rotary-percent 1.0
    --rotary-base 1000000
    --make-vocab-size-divisible-by 256
    # MoE
    --num-experts 128
    --moe-ffn-hidden-size 1536
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 8
    --moe-aux-loss-coeff 1e-3
    --attention-dropout 0.0
    --hidden-dropout 0.0
)

# ============================================================================
# MoE runtime (shared)
# ============================================================================
MOE_RUNTIME_ARGS=(
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-router-fusion
    --moe-router-dtype fp32
)

# ============================================================================
# Training config (shared)
# ============================================================================
TRAINING_ARGS=(
    --use-mcore-models
    --use-flash-attn
    --transformer-impl transformer_engine
    --train-samples 268554688
    --exit-duration-in-mins 230
    --no-create-attention-mask-in-dataloader
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --manual-gc
    --manual-gc-interval 5
    --lr 3.9e-6
    --min-lr 3.9e-7
    --lr-warmup-init 3.9e-7
    --lr-decay-style cosine
    --lr-decay-samples 584765624
    --lr-warmup-samples 1536000
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --init-method-std 0.02
    --eval-iters 32
    --eval-interval 500
    --save "${OUTPUT_DIR}/checkpoints"
    --save-interval 10000
    --enable-experimental
)

# ============================================================================
# Data config (shared — mock data)
# ============================================================================
DATA_ARGS=(
    --mock-data
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model Qwen/Qwen3-235B-A22B
    --moe-router-force-load-balancing
)

# ============================================================================
# Logging (shared)
# ============================================================================
LOGGING_ARGS=(
    --log-throughput
    --log-interval 1
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-num-zeros-in-grad
    --log-params-norm
    --log-validation-ppl-to-tensorboard
    --logging-level 40
    --tensorboard-dir "${OUTPUT_DIR}/tensorboard"
)

# ============================================================================
# Experiment-specific parallelism & batch config
# ============================================================================
case "${EXPERIMENT}" in

    fsdp_mbs1_moe_recomp)
        # FSDP EP=8, MBS=1, GBS=64, moe+layernorm recomp — 527 TFLOP/s/GPU
        SEQ_LENGTH=7550
        PARALLEL_ARGS=(
            --tensor-model-parallel-size 1
            --pipeline-model-parallel-size 1
            --expert-model-parallel-size 8
            --context-parallel-size 1
            --expert-tensor-parallel-size 1
            --sequence-parallel
            --use-megatron-fsdp
            --data-parallel-sharding-strategy optim_grads_params
            --no-gradient-accumulation-fusion
            --use-distributed-optimizer
            --ckpt-format fsdp_dtensor
            --init-model-with-meta-device
            --calculate-per-token-loss
            --fsdp-double-buffer
        )
        BATCH_ARGS=(--micro-batch-size 1 --global-batch-size 64)
        RECOMP_ARGS=(--recompute-granularity selective --recompute-modules moe layernorm)
        ;;

    fsdp_mbs1_moeact_recomp)
        # FSDP EP=8, MBS=1, GBS=64, moe_act+layernorm recomp — 560 TFLOP/s/GPU
        SEQ_LENGTH=7550
        PARALLEL_ARGS=(
            --tensor-model-parallel-size 1
            --pipeline-model-parallel-size 1
            --expert-model-parallel-size 8
            --context-parallel-size 1
            --expert-tensor-parallel-size 1
            --sequence-parallel
            --use-megatron-fsdp
            --data-parallel-sharding-strategy optim_grads_params
            --no-gradient-accumulation-fusion
            --use-distributed-optimizer
            --ckpt-format fsdp_dtensor
            --init-model-with-meta-device
            --calculate-per-token-loss
            --fsdp-double-buffer
        )
        BATCH_ARGS=(--micro-batch-size 1 --global-batch-size 64)
        RECOMP_ARGS=(--recompute-granularity selective --recompute-modules moe_act layernorm)
        ;;

    fsdp_mbs2_moe_recomp)
        # FSDP EP=8, MBS=2, GBS=128, moe+layernorm recomp — 640 TFLOP/s/GPU
        SEQ_LENGTH=7550
        PARALLEL_ARGS=(
            --tensor-model-parallel-size 1
            --pipeline-model-parallel-size 1
            --expert-model-parallel-size 8
            --context-parallel-size 1
            --expert-tensor-parallel-size 1
            --sequence-parallel
            --use-megatron-fsdp
            --data-parallel-sharding-strategy optim_grads_params
            --no-gradient-accumulation-fusion
            --use-distributed-optimizer
            --ckpt-format fsdp_dtensor
            --init-model-with-meta-device
            --calculate-per-token-loss
            --fsdp-double-buffer
        )
        BATCH_ARGS=(--micro-batch-size 2 --global-batch-size 128)
        RECOMP_ARGS=(--recompute-granularity selective --recompute-modules moe layernorm)
        ;;

    pp8_vpp4_mbs1)
        # PP8/VPP4/EP8, MBS=1, GBS=64, moe_act+layernorm recomp — 645 TFLOP/s/GPU
        SEQ_LENGTH=7550
        PARALLEL_ARGS=(
            --tensor-model-parallel-size 1
            --pipeline-model-parallel-size 8
            --num-layers-per-virtual-pipeline-stage 3
            --expert-model-parallel-size 8
            --context-parallel-size 1
            --expert-tensor-parallel-size 1
            --sequence-parallel
            --account-for-embedding-in-pipeline-split
            --account-for-loss-in-pipeline-split
            --use-distributed-optimizer
            --overlap-grad-reduce
            --overlap-param-gather
        )
        BATCH_ARGS=(--micro-batch-size 1 --global-batch-size 64)
        RECOMP_ARGS=(--recompute-granularity selective --recompute-modules moe_act layernorm)
        ;;

    fsdp_cp4_seq26k)
        # FSDP CP=4/EP=8, MBS=1, GBS=16, no recomp, seq=26464 — 645 TFLOP/s/GPU
        SEQ_LENGTH=26464
        PARALLEL_ARGS=(
            --tensor-model-parallel-size 1
            --pipeline-model-parallel-size 1
            --expert-model-parallel-size 8
            --context-parallel-size 4
            --cp-comm-type a2a
            --expert-tensor-parallel-size 1
            --sequence-parallel
            --use-megatron-fsdp
            --data-parallel-sharding-strategy optim_grads_params
            --no-gradient-accumulation-fusion
            --use-distributed-optimizer
            --ckpt-format fsdp_dtensor
            --init-model-with-meta-device
            --calculate-per-token-loss
            --fsdp-double-buffer
        )
        BATCH_ARGS=(--micro-batch-size 1 --global-batch-size 16)
        RECOMP_ARGS=()
        ;;

    pp8_vpp4_cp4_seq26k)
        # PP8/VPP4/CP4/EP8, MBS=1, GBS=16, no recomp, seq=26464 — 676 TFLOP/s/GPU
        SEQ_LENGTH=26464
        PARALLEL_ARGS=(
            --tensor-model-parallel-size 1
            --pipeline-model-parallel-size 8
            --num-layers-per-virtual-pipeline-stage 3
            --expert-model-parallel-size 8
            --context-parallel-size 4
            --cp-comm-type a2a
            --expert-tensor-parallel-size 1
            --sequence-parallel
            --account-for-embedding-in-pipeline-split
            --account-for-loss-in-pipeline-split
            --use-distributed-optimizer
            --overlap-grad-reduce
            --overlap-param-gather
        )
        BATCH_ARGS=(--micro-batch-size 1 --global-batch-size 16)
        RECOMP_ARGS=()
        ;;

    *)
        echo "Unknown EXPERIMENT: ${EXPERIMENT}"
        echo "Available: fsdp_mbs1_moe_recomp, fsdp_mbs1_moeact_recomp, fsdp_mbs2_moe_recomp,"
        echo "           pp8_vpp4_mbs1, fsdp_cp4_seq26k, pp8_vpp4_cp4_seq26k"
        exit 1
        ;;
esac

DATA_ARGS+=(--seq-length "${SEQ_LENGTH}")

# ============================================================================
# Environment variables
# ============================================================================
ENV_EXPORTS=(
    "export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1"
    "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    "export NCCL_NVLS_ENABLE=0"
    "export NCCL_GRAPH_REGISTER=0"
    "export NVTE_FUSED_ATTN=1"
    "export NVTE_NORM_FWD_USE_CUDNN=1"
    "export NVTE_NORM_BWD_USE_CUDNN=1"
    "export TRITON_CACHE_DIR=\${TRITON_CACHE_DIR:-/tmp/triton_cache_\${SLURM_NODEID}}"
    "export HF_HOME=/lustre/fsw/coreai_devtech_all/\${USER}/.cache/huggingface"
)

# ============================================================================
# Build sbatch script
# ============================================================================
JOB_NAME="qwen3-dit-${EXPERIMENT}-$(date +%y%m%d_%H%M%S)"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

TRAINING_SCRIPT="${MEGATRON_PATH}/pretrain_gpt.py"

ALL_ARGS=(
    "${MODEL_ARGS[@]}"
    "${DATA_ARGS[@]}"
    "${PARALLEL_ARGS[@]}"
    "${MOE_RUNTIME_ARGS[@]}"
    "${BATCH_ARGS[@]}"
    "${RECOMP_ARGS[@]}"
    "${TRAINING_ARGS[@]}"
    "${LOGGING_ARGS[@]}"
)

SBATCH_SCRIPT=$(mktemp /tmp/qwen3_dit_XXXXXX.sh)
cat > "${SBATCH_SCRIPT}" <<SBATCH_EOF
#!/bin/bash
#SBATCH --nodes=${NNODES}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --ntasks-per-node=1
#SBATCH --time=${TIME_LIMIT}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.log
#SBATCH --exclusive

set -euxo pipefail

# Environment
$(printf '%s\n' "${ENV_EXPORTS[@]}")

export MASTER_ADDR=\$(scontrol show hostname \${SLURM_NODELIST} | head -n1)

# Training command
TRAINING_CMD="python -m torch.distributed.run \\
    --nproc_per_node=${GPUS_PER_NODE} \\
    --nnodes=${NNODES} \\
    --master_addr=\\\$MASTER_ADDR \\
    --master_port=6000 \\
    --node_rank=\\\$SLURM_NODEID \\
    ${TRAINING_SCRIPT} \\
    ${ALL_ARGS[*]}"

srun \\
    --mpi=pmix -l \\
    --export=ALL \\
    --no-container-mount-home \\
    --container-image=${CONTAINER_IMAGE} \\
    --container-mounts=${CONTAINER_MOUNTS} \\
    --container-workdir=${MEGATRON_PATH} \\
    bash -c "\${TRAINING_CMD}"
SBATCH_EOF

echo "============================================================"
echo "Experiment: ${EXPERIMENT}"
echo "Nodes: ${NNODES}, GPUs/node: ${GPUS_PER_NODE}"
echo "SeqLength: ${SEQ_LENGTH}"
echo "Script: ${SBATCH_SCRIPT}"
echo "============================================================"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY RUN — sbatch script:"
    cat "${SBATCH_SCRIPT}"
else
    sbatch "${SBATCH_SCRIPT}"
fi
