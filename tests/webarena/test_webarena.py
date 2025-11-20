import sys
import os
from pathlib import Path

# Add workspace root to Python path for imports
workspace_root = Path(__file__).parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from agent_system.environments.env_manager import WebVoyagerEnvironmentManager
from agent_system.environments.env_package.webvoyager import build_webvoyager_envs, webvoyager_projection
import types

os.environ['WEBARENA_DATA'] = '/usr3/graduate/xfl/lab/verl-agent/agent_system/environments/env_package/webvoyager/webvoyager/data'
config_path = os.path.join(os.path.dirname(__file__), '../..', 'agent_system/environments/env_package/webvoyager/configs/webarena_configs.yaml')
resources_per_worker = {"num_cpus": 0.1, "num_gpus": 0}
_envs = build_webvoyager_envs(config_path, seed=1, env_num=16, group_n=4, resources_per_worker=resources_per_worker, is_train=True)
class _EnvNS(types.SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key) if hasattr(self, key) else default

config = types.SimpleNamespace()
env_manager = WebVoyagerEnvironmentManager(_envs, webvoyager_projection, config)

obs, info = env_manager.reset(kwargs={})
print(obs)