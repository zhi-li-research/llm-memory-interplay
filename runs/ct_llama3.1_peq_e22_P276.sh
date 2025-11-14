#!/usr/bin/env bash

#SBATCH --partition=alvis
#SBATCH --account=NAISS2024-22-202
#SBATCH --nodes=1
#SBATCH --time=3-24:00:00
#SBATCH --job-name=ct_llama3.1_peq_e22_P276
#SBATCH --error=./logs/ct_llama3.1_peq_e22_P276-%J.err.log
#SBATCH --output=./logs/ct_llama3.1_peq_e22_P276-%J.out.log
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=A100:1

set -eo pipefail

module load Python/3.9.5-GCCcore-10.3.0
module load CUDA/11.3.1

export CUDA_LAUNCH_BLOCKING=1
export PROJECT_DIR=/mimer/NOBACKUP/groups/naiss2024-23-531/zhi/llm-memory-interplay
export CACHE_DIR=/mimer/NOBACKUP/groups/naiss2024-23-531/.cache

# Change these blocks based on your environment
export WANDB_CACHE_DIR="${CACHE_DIR}/wandb"
export TRANSFORMERS_CACHE="${CACHE_DIR}/huggingface/transformers"
export HF_HOME="${CACHE_DIR}/huggingface/transformers"
export HF_DATASETS_CACHE="${CACHE_DIR}/huggingface/datasets"

mkdir -p "${WANDB_CACHE_DIR}"
mkdir -p "${TRANSFORMERS_CACHE}"
mkdir -p "${HF_DATASETS_CACHE}"

# Change these blocks based on your environment
source "${PROJECT_DIR}/venv/bin/activate"
echo "PYTHON PATH:"
which python

MODEL_NAME="/mimer/NOBACKUP/groups/naiss2024-23-531/mehrdad/models/Llama-3.1-8B"
PROMPT_FORMAT="C: {context} {prompt}"

DATA_PATH=./data/data/peq/both/P276.jsonl
OUTPUT_DIR=./experiments/ct/llama/peq/e22/P276

EXPERIMENT_TYPE=3
MAX_DATAPOINTS=100
REVERSE_PATCHING=0
REPLACE=0
WINDOW=10
SAMPLES=10
MAX_CF=3

echo "EXPERIMENT (${EXPERIMENT_TYPE})"
# srun echo $PWD
srun python src/run_causal_tracer.py \
    --model_name="$MODEL_NAME" \
    --experiment_type=$EXPERIMENT_TYPE \
    --data_path="$DATA_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --reverse_patching=$REVERSE_PATCHING \
    --replace=$REPLACE \
    --samples=$SAMPLES \
    --max_datapoints=$MAX_DATAPOINTS \
    --window=$WINDOW \
    --max_cf=$MAX_CF \
    --prompt_format="$PROMPT_FORMAT"