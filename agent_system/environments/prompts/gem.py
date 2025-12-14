"""Prompt templates for GEM environments.

GEM environments are text-based games/tasks. Most tasks require that the final action
contains a command formatted inside \\boxed{...}. We keep the same `<think>` and
`<action>` protocol as the rest of verl-agent.
"""

# --------------------- GEM --------------------- #

GEM_TEMPLATE_MULTI_EPISODE_INIT = """
You are an expert agent operating in GEM, a suite of interactive text-based games and tasks.
Every episode resets on: success or after {episode_cap} step(s). The game/task configuration is unchanged across episodes, so you can reuse what you learned from previous episodes.

Game/task id: {env_id}
You are currently in episode 1 and your current observation is:
{current_observation}
{task_suffix}

Now it's your turn to take an action.
- You MUST first reason step-by-step in <think>...</think>.
- Then you MUST output the action to send to the environment inside <action>...</action>.
- The <action> content MUST follow the game's required format (usually includes a \\boxed{{...}} command).
- In <action>, output EXACTLY ONE \\boxed{{...}} command and nothing else (no extra words, no multiple boxed actions).

Output format example (follow this exactly):
<think>...</think>
<action>\\boxed{{up}}</action>
"""

GEM_TEMPLATE_MULTI_EPISODE = """
You are an expert agent operating in GEM, a suite of interactive text-based games and tasks.
Every episode resets on: success or after {episode_cap} step(s). The game/task configuration is unchanged across episodes, so you can reuse what you learned from previous episodes.

Game/task id: {env_id}
Prior to this step, you have already taken {step_count} step(s) across all episodes.
Below are the most recent {history_length} observations and the corresponding actions you took, including episode boundaries and results if applicable:
{action_history}

You are currently in episode {current_episode}, step {current_step}.
Current observation:
{current_observation}
{task_suffix}

Now it's your turn to take an action.
- You MUST first reason step-by-step in <think>...</think>.
- Then you MUST output the action to send to the environment inside <action>...</action>.
- The <action> content MUST follow the game's required format (usually includes a \\boxed{{...}} command).
- In <action>, output EXACTLY ONE \\boxed{{...}} command and nothing else (no extra words, no multiple boxed actions).

Output format example (follow this exactly):
<think>...</think>
<action>\\boxed{{up}}</action>
"""

# Backward-compatible alias used by early experiments.
GEM_INIT_PROMPT = GEM_TEMPLATE_MULTI_EPISODE_INIT
