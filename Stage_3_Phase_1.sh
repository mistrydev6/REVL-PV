#!/bin/bash

export WORKSPACE_DIR="$(pwd)"
export DATASET_PATH="PATH TO YOUR DATASET" 
export PRETRAIN_MODEL_PATH="PATH TO YOUR MODEL" 
export SAVE_PATH="${WORKSPACE_DIR}/checkpoints"
export OPENRLHF_FRE_TEXT=1
export MODEL_NAME="NAME OF THE TRAINING RUN" 

export WANDB_DIR="${WORKSPACE_DIR}"
export WANDB_API_KEY="WANDB API KEY"

export no_proxy="127.0.0.1,localhost,::1"
export NO_PROXY="127.0.0.1,localhost,::1"


SCRIPT_PID=$$
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export LOG_DIR="${SAVE_PATH}/${MODEL_NAME}/logs"
export CUR_LOG_DIR="${LOG_DIR}/${TIMESTAMP}"

ray stop

mkdir -p "${SAVE_PATH}/${MODEL_NAME}"
mkdir -p "${LOG_DIR}"
mkdir -p "${CUR_LOG_DIR}"


echo "================================================================"
echo "2SRE FRE-Text Training"
echo "================================================================"
echo "Model name: ${MODEL_NAME}"
echo "Dataset: ${DATASET_PATH}"
echo "Pretrained model: ${PRETRAIN_MODEL_PATH}"
echo "Logs will be saved to: ${CUR_LOG_DIR}"
echo
echo "To monitor logs:"
echo "  tail -f ${CUR_LOG_DIR}/train.log"
echo
echo "================================================================"


echo "Starting ray..."
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8 --temp-dir TEMP DIR --dashboard-port 8266

echo "Waiting for Ray to initialize..."
sleep 60 

ray status

echo "Checking dashboard availability..."
curl -s http://127.0.0.1:8266/api/version || echo "Warning: Dashboard not responding yet."

echo "Killing any existing process on port 5000..."
fuser -k 5000/tcp || true

echo "Starting remote reward model server on default port 5000..."
python /eagle/PBML/mistrydev/lmm/lmm-r1/openrlhf/models/remote_rm/solar_verifier.py &
REMOTE_RM_PID=$!

sleep 10

echo "Starting training..."
MAX_RETRIES=3
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    ray job submit --address="http://127.0.0.1:8266" \
   --runtime-env-json='{"working_dir": "'${WORKSPACE_DIR}'","env_vars":{"VLLM_USE_V1":"1","VLLM_ENABLE_V1_MULTIPROCESSING":"0","OPENRLHF_FRE_TEXT":"1","VLLM_SKIP_P2P_CHECK":"1"}, "excludes":["checkpoints","wandb",".git"]}' \
   -- python -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --remote_rm_url http://127.0.0.1:5000/get_reward \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 8 \
   --vllm_num_engines 8 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_enable_sleep \
   --vllm_gpu_memory_utilization 0.45 \
   --vllm_sync_backend nccl \
   --enable_prefix_caching \
   --enforce_eager \
   --pretrain ${PRETRAIN_MODEL_PATH} \
   --save_path ${SAVE_PATH}/${MODEL_NAME} \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 64 \
   --temperature 1.0 \
   --n_samples_per_prompt 6 \
   --max_epochs 1 \
   --num_episodes 2 \
   --prompt_max_len 2048 \
   --max_samples 9500 \
   --generate_max_len 2048 \
   --advantage_estimator rloo \
    --zero_stage 3 \
    --adam_offload \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --init_kl_coef 0.001 \
   --prompt_data ${DATASET_PATH} \
   --input_key message \
   --label_key "answer" \
   --normalize_reward \
   --flash_attn \
   --lambd 1 \
   --gamma 1 \
   --gradient_checkpointing \
   --save_steps 100 \
   --ckpt_path ${SAVE_PATH}/${MODEL_NAME}/ckpt \
   --save_hf_ckpt \
   --load_checkpoint \
   --packing_samples \
   --use_tensorboard ${LOG_DIR}


    TRAIN_PID=$!
    if [ ! -z "$TRAIN_PID" ] && ps -p $TRAIN_PID > /dev/null; then
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT+1))
    echo "Job submission failed (attempt $RETRY_COUNT/$MAX_RETRIES). Retrying in 30 seconds..."
    sleep 30
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Job submission failed after $MAX_RETRIES attempts. Check Ray logs in /eagle/PBML/mistrydev/ray_temp/session_latest/logs."
    exit 1
fi

echo "Remote RM PID: $REMOTE_RM_PID" > "${CUR_LOG_DIR}/process_pids.txt"
echo "Train PID: $TRAIN_PID" >> "${CUR_LOG_DIR}/process_pids.txt"

echo "Training is running in the background. Check logs at ${CUR_LOG_DIR}/train.log"
echo "To attach to the training process: wait $TRAIN_PID"

echo "When finished, clean up with:"
echo "pkill -f openrlhf"
echo "ray stop"
echo "All logs are available in ${CUR_LOG_DIR}"

