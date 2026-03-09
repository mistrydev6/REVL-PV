#!/bin/bash

export WORKSPACE_DIR="$(pwd)"
export DATASET_PATH="PATH TO YOUR DATASET" 
export PRETRAIN_MODEL_PATH="PATH TO YOUR STAGE 2 PHASE 1 CHECKPOINT" 
export SAVE_PATH="${WORKSPACE_DIR}/checkpoints"
export MODEL_NAME="NAME OF THE TRAINING RUN" 

export no_proxy="127.0.0.1,localhost,::1"
export NO_PROXY="$no_proxy"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_TORCH_COMPILE=0
export NCCL_P2P_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1

export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export LOG_DIR="${SAVE_PATH}/${MODEL_NAME}/logs"
export CUR_LOG_DIR="${LOG_DIR}/${TIMESTAMP}"
mkdir -p "${SAVE_PATH}/${MODEL_NAME}" "${LOG_DIR}" "${CUR_LOG_DIR}"

cleanup() {
  echo "Cleaning up..."
  if [[ -n "${REMOTE_RM_PID:-}" ]]; then kill "${REMOTE_RM_PID}" >/dev/null 2>&1 || true; fi
  ray stop >/dev/null 2>&1 || true
}
trap cleanup EXIT

ray stop >/dev/null 2>&1 || true
echo "Starting Ray (dashboard: 8266)..."
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8 --temp-dir ~/.cache/ray --dashboard-port 8266

echo "Waiting for Ray to initialize..."
sleep 5
ray status || true

echo "Starting remote reward model server on default port 5000..."
python /eagle/PBML/mistrydev/lmm/lmm-r1/openrlhf/models/remote_rm/solar_verifier.py &
REMOTE_RM_PID=$!

echo "Waiting for reward server on 127.0.0.1:5000/get_reward ..."
for i in {1..60}; do
  code=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "http://127.0.0.1:5000/get_reward" \
    -H 'Content-Type: application/json' \
    -d '{"query":["ping"],"prompts":["p"],"labels":["l"]}' || echo 000)
  if [[ "$code" == "200" || "$code" == "422" ]]; then
    echo "Reward server is up (HTTP ${code})."
    break
  fi
  if ! ps -p ${REMOTE_RM_PID} >/dev/null 2>&1; then
    echo "Reward server exited early. Check logs: ${CUR_LOG_DIR}/remote_rm.log"
    exit 1
  fi
  sleep 1
done

RUNTIME_ENV_JSON=$(printf \
'{"working_dir":"%s","excludes":["checkpoints","wandb",".git"],"env_vars":{"PYTORCH_CUDA_ALLOC_CONF":"%s","VLLM_USE_V1":"%s","VLLM_ENABLE_V1_MULTIPROCESSING":"%s","VLLM_TORCH_COMPILE":"%s"}}' \
"$WORKSPACE_DIR" "$PYTORCH_CUDA_ALLOC_CONF" "$VLLM_USE_V1" "$VLLM_ENABLE_V1_MULTIPROCESSING" "$VLLM_TORCH_COMPILE")

ray job submit --address="http://127.0.0.1:8266" \
  --runtime-env-json="$RUNTIME_ENV_JSON" \
  -- python -m openrlhf.cli.train_ppo_ray \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 1 \
  --remote_rm_url http://127.0.0.1:5000/get_reward \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 4 \
  --critic_num_nodes 1 \
  --critic_num_gpus_per_node 2 \
  --vllm_num_engines 1 \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.6 \
  --vllm_enable_sleep \
  --vllm_sync_backend nccl \
  --enable_prefix_caching \
  --adam_offload \
  --pretrain "${PRETRAIN_MODEL_PATH}" \
  --save_path "${SAVE_PATH}/${MODEL_NAME}" \
  --micro_train_batch_size 1 \
  --train_batch_size 16 \
  --micro_rollout_batch_size 1 \
  --rollout_batch_size 16 \
  --n_samples_per_prompt 1 \
  --temperature 1.0 \
  --max_epochs 1 \
  --num_episodes 1 \
  --prompt_max_len 2048 \
  --max_samples 100000 \
  --generate_max_len 512 \
  --advantage_estimator gae \
  --zero_stage 3 \
  --bf16 \
  --actor_learning_rate 1e-6 \
  --init_kl_coef 0.001 \
  --prompt_data "${DATASET_PATH}" \
  --input_key message \
  --label_key "answer" \
  --flash_attn \
  --lambd 1 \
  --gamma 1 \
  --gradient_checkpointing \
  --save_steps 20 \
  --ckpt_path "${SAVE_PATH}/${MODEL_NAME}/ckpt" \
  --save_hf_ckpt \
  --load_checkpoint \
  --use_wandb false \
  --use_tensorboard "${LOG_DIR}"

TRAIN_PID=$!
echo "Remote RM PID: $REMOTE_RM_PID" > "${CUR_LOG_DIR}/process_pids.txt"
echo "Train PID: $TRAIN_PID" >> "${CUR_LOG_DIR}/process_pids.txt"

echo "================================================================"
echo "Training submitted with PID: $TRAIN_PID"
echo "Logs: ${CUR_LOG_DIR}"
echo "================================================================"

wait $TRAIN_PID