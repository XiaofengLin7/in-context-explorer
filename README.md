## Installation
Please refer to README-verl-agent.md
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