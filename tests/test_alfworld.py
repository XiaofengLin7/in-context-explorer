from agent_system.environments.env_manager import AlfWorldEnvironmentManager
from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
import os
import types

os.environ['ALFWORLD_DATA'] = '/projectnb/replearn/xfl/alfworld/data_storage'
# Update path to work from tests/ directory (go up one level to project root)
alf_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'agent_system/environments/env_package/alfworld/configs/config_tw.yaml'))

resources_per_worker = {"num_cpus": 0.1, "num_gpus": 0}
_envs = build_alfworld_envs(alf_config_path, seed=1, env_num=1, group_n=1, resources_per_worker=resources_per_worker, is_train=False)

# Create config object with SimpleNamespace structure
class _EnvNS(types.SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key) if hasattr(self, key) else default

config = types.SimpleNamespace()
env = _EnvNS()
env.prompt_type = 'gold'
env.history_length = 1
config.env = env


env_manager = AlfWorldEnvironmentManager(_envs, alfworld_projection, config)

obs, info = env_manager.reset(kwargs={})
print(env_manager.receptacles)
print(obs['text'])
text_actions = ["<action>go to dresser 1</action>"]
next_obs, rewards, dones, infos = env_manager.step(text_actions)
print(next_obs['text'])

assert env_manager.visited_receptacles[0] == {"dresser 1"}
text_actions = ["<action>go to shelf 5</action>"]
next_obs, rewards, dones, infos = env_manager.step(text_actions)
print(next_obs['text'])
assert env_manager.visited_receptacles[0] == {"dresser 1", "shelf 5"}
text_actions = ["<action>go to drawer 1</action>"]
next_obs, rewards, dones, infos = env_manager.step(text_actions)
print(next_obs['text'])
assert env_manager.visited_receptacles[0] == {"dresser 1", "shelf 5"}
text_actions = ["<action>open drawer 1</action>"]
next_obs, rewards, dones, infos = env_manager.step(text_actions)
print(next_obs['text'])
assert env_manager.visited_receptacles[0] == {"dresser 1", "shelf 5", "drawer 1"}
text_actions = ["<action>go to garbagecan 1</action>"]
next_obs, rewards, dones, infos = env_manager.step(text_actions)
print(next_obs['text'])
assert env_manager.visited_receptacles[0] == {"dresser 1", "shelf 5", "drawer 1", "garbagecan 1"}

