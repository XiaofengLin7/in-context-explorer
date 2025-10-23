## Installation
For alfworld installation, please refer to README-verl-agent.md.
### Webvoyager
```
git submodule update --init --recursive
conda activate verl-agent # activate your previous installed conda env following README-verl-agent.md
pip install -r agent_system/environments/env_package/webvoyager/webvoyager/requirements.txt
```
Test your webvoyager
```
cd agent_system/environments/env_package/webvoyager/webvoyager
bash run.sh
```
check your /results/your_exp_time/agent.log under current webvoyager directory, if makes sense, then configuration is finished.
## Experiments
### Training scripts
configure your ALFWORLD_DATA first and your desired number of gpus in this training script first.
```
bash examples/grpo_trainer/run_alfworld.sh
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