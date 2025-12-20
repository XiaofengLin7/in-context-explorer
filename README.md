## Installation
For alfworld installation, please refer to README-verl-agent.md.
### Webvoyager
```
git submodule update --init --recursive
conda activate verl-agent # activate your previous installed conda env following README-verl-agent.md
pip install -r agent_system/environments/env_package/webvoyager/webvoyager/requirements.txt
pip install selenium==4.15.2
pip install anthropic
pip install nltk
```
Test your webvoyager
```
cd agent_system/environments/env_package/webvoyager/webvoyager
bash run.sh
```
check your /results/your_exp_time/agent.log under current webvoyager directory, if makes sense, then configuration is finished.
Please refer to README-verl-agent.md
### Appworld
```
conda create -n verl-agent-appworld python=3.12 -y
conda activate verl-agent-appworld
pip install git+https://github.com/StonyBrookNLP/appworld.git
appworld install
appworld download data
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.5
pip3 install ray==2.49.0
```

### GEM
```
conda activate verl-agent
pip install -e agent_system/environments/env_package/gem/gem
```

#### Test your GEM
run cell in agent_system/environments/env_package/gem/gem_demo.ipynb

#### Soft reset / multi-episode rollout semantics (GEM)
GEM training commonly uses **multi-episode rollout**: a single rollout trajectory is split into multiple
"episodes" on the **same task instance**, without clearing the agent's history buffer. This is implemented
via `soft_reset`.

**What triggers a `soft_reset` (when `env.multi_episode_rollout.enable=True`)**
- **Success**: when the env ends and `info["won"] == True` (GEM uses terminal "Congratulations!" messages).
- **Episode step limit**: when the per-episode step counter reaches `env.multi_episode_rollout.episode_max_steps`.
- **Internal max turns**: when a GEM game ends due to its inherent `max_turns` (the env truncates with a message
  containing "maximum number of turns"). We tag this as `info["internal_max_turns"] == True` and soft-reset.
- **Any other terminal** (GEM only): format errors, failures, etc. also trigger a soft reset (reason: `"terminal"`).
  This keeps the trajectory length consistent (i.e. `episode_length` reaches `env.max_steps`) while still letting the
  policy learn from failures.

**Important interaction of step limits**
- Each GEM game has its own `max_turns` (e.g. Minesweeper-easy: 25). Our code also supports an outer cap
  `env.max_steps` (trajectory length) and `episode_max_steps` (episode length within the trajectory).
- If you want soft resets to happen *because of internal max turns*, set:
  `env.multi_episode_rollout.episode_max_steps >= max(max_turns)` across your task pool.

**What happens on `soft_reset`**
- The environment resets to the **same (env_id, seed)** task instance.
- The agent memory/history is **not** cleared; episode boundary/result text is appended into history.

#### Single-task GEM training (no extra config files)
If you want to train on just one GEM game and keep your launch clean, disable the default multi-task pool and set a single env id. Seeds will be sampled automatically on each reset, so no task files are needed.

- GuessTheNumber (random seeds each reset)
  ```bash
  python3 -m verl.trainer.main_ppo \
    ... \
    env.env_name=gem \
    +env.gem.use_default_pool=False \
    +env.gem.env_ids=["game:GuessTheNumber-v0-easy"]
  ```

- Minesweeper (random seeds each reset)
  ```bash
  python3 -m verl.trainer.main_ppo \
    ... \
    env.env_name=gem \
    +env.gem.use_default_pool=False \
    +env.gem.env_ids=["game:Minesweeper-v0-easy"]
  ```

Optional: fixed seeds for reproducibility without extra files
```bash
python3 -m verl.trainer.main_ppo \
  ... \
  env.env_name=gem \
  +env.gem.use_default_pool=False \
  +env.gem.env_ids=["game:GuessTheNumber-v0-easy"] \
  +env.gem.task_pool_train=[{env_id:"game:GuessTheNumber-v0-easy",seed:1},{env_id:"game:GuessTheNumber-v0-easy",seed:2}]
```

#### Troubleshooting (common environment issues)

##### 1) OmegaConf import error: `Could not deserialize ATN with version 3 (expected 4)`
**Symptoms**
- Importing `OmegaConf` / running training crashes when importing `agent_system/environments/env_manager.py` with:
  - `Exception: Could not deserialize ATN with version 3 (expected 4).`

**Cause**
- `omegaconf==2.3.0` expects an older ANTLR runtime, but your environment has a newer
  `antlr4-python3-runtime` installed (e.g., 4.13.x).

**Fix (recommended for `omegaconf==2.3.0`)**
```bash
conda activate verl-agent
pip install --no-deps --force-reinstall "antlr4-python3-runtime==4.9.3"
python -c "from omegaconf import OmegaConf; print(OmegaConf.create({'ok': True}).ok)"
```

##### 2) Wordle/Hangman fails under Ray due to NLTK download races
**Symptoms**
- When initializing GEM Wordle/Hangman in parallel (Ray workers), you may see errors like:
  - `FileExistsError: .../nltk_data/corpora/words`
  - repeated `nltk.download("words")` messages from many workers

**Cause**
- GEM's Wordle/Hangman envs call `nltk.download("words")` in `__init__`, and concurrent
  downloads can race.

**Fix**
- Pre-download the corpus once and/or set a stable cache directory:
```bash
conda activate verl-agent
export NLTK_DATA="$HOME/.cache/nltk_data"
python -c "import nltk; nltk.download('words', download_dir='$HOME/.cache/nltk_data')"
```
Note: this repo also adds a lightweight lock in the GEM wrapper to avoid concurrent
downloads, but pre-downloading is still recommended on shared machines.
## Experiments
### Training scripts
configure your ALFWORLD_DATA first and your desired number of gpus in this training script first.
```
bash examples/grpo_trainer/run_alfworld.sh
```

```
# start appworld server
bash examples/env_server/start_appworld_server.sh
# on another terminal
bash examples/gigpo_trainer/run_appworld.sh
```

### Tune lambda/success_coef
if success_coef==0, then orginal reward will be used.
```script
    env.success_coef=10.0
```
### Prompt selection
summary will prompt the agent to output known and unknown; vanilla is the original prompt version.
```script
    env.prompt_type=summary  #or vanilla
```
