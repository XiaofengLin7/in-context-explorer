set -x

# ── Activate lamer venv (mirrors lamer/scripts/eval_alfworld.sh setup) ────────
source /home/jobuser/.venv/lamer/bin/activate

# ── Repo root: where openconnect uploads the resources ───────────────────────
ICE_ROOT=/home/jobuser/resources
export PYTHONPATH="${ICE_ROOT}:${PYTHONPATH}"
cd "${ICE_ROOT}"

export ALFWORLD_DATA=${ALFWORLD_DATA:-/home/jobuser/.cache/alfworld}
N_GPUS=${N_GPUS:-8}
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.
model_path=${model_path:-/shared/public/elr-models/Qwen/Qwen3-8B/2069b3fae1114555f3c020c81410e51fa0f656f2_130k_context}
train_data_size=16
val_data_size=128
group_size=8
prompt_type=chat
history_length=30
env_max_steps=10
experiment_name=grpo_${model_path}_prompt_type_${prompt_type}_history_length_${history_length}_env_max_steps_${env_max_steps}

# We only use data preparation to indicate the modality and the data size.
# Create dummy scaffold parquets without downloading from HuggingFace
# (mirrors lamer/scripts/eval_alfworld.sh — no internet in cluster)
DATA_DIR="${HOME}/data/verl-agent/text"
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    mkdir -p "${DATA_DIR}"
    python3 - <<'EOF'
import os, pandas as pd
data_dir = os.path.join(os.environ["HOME"], "data/verl-agent/text")
os.makedirs(data_dir, exist_ok=True)
def make_rows(split, n):
    return [{"data_source": "text",
             "prompt": [{"role": "user", "content": ""}],
             "ability": "agent",
             "extra_info": {"split": split, "index": i}} for i in range(n)]
pd.DataFrame(make_rows("train", int(os.environ.get("train_data_size", 16)))).to_parquet(os.path.join(data_dir, "train.parquet"))
pd.DataFrame(make_rows("test",  int(os.environ.get("val_data_size",   256)))).to_parquet(os.path.join(data_dir, "test.parquet"))
print("Created dummy scaffold data at", data_dir)
EOF
fi

echo "=== Sanity check: workflow args ==="
echo "model_path=${model_path}"
echo "ENABLE_REFLECTION=${ENABLE_REFLECTION:-False}"
echo "==================================="

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=28670 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.prompt_type=$prompt_type \
    env.max_steps=30 \
    env.rollout.n=$group_size \
    env.history_length=$history_length \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.multi_episode_rollout.enable=True \
    env.multi_episode_rollout.reward_per_completion=1.0 \
    env.multi_episode_rollout.episode_max_steps=$env_max_steps \
    +env.multi_episode_rollout.enable_reflection=${ENABLE_REFLECTION:-False} \
    +env.verbose_rollout=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','mlflow'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=300 \
    trainer.test_freq=5 \
    trainer.total_epochs=0 \
    trainer.val_before_train=True $@
