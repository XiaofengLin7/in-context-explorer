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
