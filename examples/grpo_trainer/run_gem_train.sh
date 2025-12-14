set -x

N_GPUS=4
ENGINE=vllm

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Stable NLTK cache directory (Wordle/Hangman in eval pool).
export NLTK_DATA=${NLTK_DATA:-"$HOME/.cache/nltk_data"}

num_cpus_per_env_worker=0.1

# Data sizes: these only control the number of prompts (env batch size).
# Keep defaults aligned with alfworld multi-episode example for a "standard" run.
train_data_size=16
val_data_size=200

# GRPO group size (number of sampled responses per prompt).
group_size=8

# Prompt/history.
history_length=30

# Trajectory settings:
# - env.max_steps: total steps collected per rollout trajectory
# - env.multi_episode_rollout.episode_max_steps: soft-reset every K steps (or on success)
env_max_steps=30
episode_max_steps=10

# Model + optimization knobs
model_path=Qwen/Qwen3-1.7B

experiment_name=grpo_gem_multitask_${model_path}

python3 -m examples.data_preprocess.prepare \
  --mode 'text' \
  --train_data_size $train_data_size \
  --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/verl-agent/text/train.parquet \
  data.val_files=$HOME/data/verl-agent/text/test.parquet \
  data.train_batch_size=$train_data_size \
  data.val_batch_size=$val_data_size \
  data.max_prompt_length=10240 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path=$model_path \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=$ENGINE \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  algorithm.use_kl_in_reward=False \
  env.env_name=gem \
  env.seed=0 \
  env.max_steps=$env_max_steps \
  env.rollout.n=$group_size \
  env.history_length=$history_length \
  env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
  +env.gem.use_default_pool=True \
  env.multi_episode_rollout.enable=True \
  env.multi_episode_rollout.episode_max_steps=$episode_max_steps \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  trainer.project_name='verl_agent_gem' \
  trainer.experiment_name=$experiment_name \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=10 \
  trainer.total_epochs=300 \
  trainer.val_before_train=True $@

