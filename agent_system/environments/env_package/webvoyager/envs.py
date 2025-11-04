import ray
import gym
import numpy as np
import json
import os
import yaml
from pprint import pprint

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

class WebVoyagerWorker:
    def __init__(self, seed, data_file, env_kwargs):
        from .webgym import WebVoyagerEnv
        self.env = WebVoyagerEnv(
                    api_key="your-api-key-here",
                    headless=True,
                    text_only=False
                )
        self.data_file = data_file
        self.tasks = []
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.tasks.append(json.loads(line))

    def step(self, action):
        """Execute a step in the environment"""
        obs, reward, done, info = self.env.step(action)
        
        #info = dict(info or {})  # make a *copy* so we can mutate safely
        #info['available_actions'] = self.env.get_available_actions()
        #info['task_score'] = reward

        #if done and reward == 1.0:
        #    info['won'] = True
        #    reward = 10.0
        #else:
        #    info['won'] = False
        #    reward = 0

        return obs, reward, done, info
    
    def reset(self, idx):
        """Reset the environment with given a task dict"""
        task = self.tasks[idx]
        obs, info = self.env.reset(task=task)
        #info = dict(info or {})
        #info['available_actions'] = self.env.get_available_actions()
        #info['won'] = False
        return obs, info
    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()
    
    def close(self):
        """Close the environment"""
        self.env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebVoyagerMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        config_path: str = "configs/configs.yaml",
        env_kwargs: dict = None,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1
        self.config = load_config_file(config_path)
        raw_path = self.config["dataset"]["train_data_path"] if is_train else self.config["dataset"]["test_data_path"]
        # Expand environment variables like $WEBVOYAGER_DATA in the path
        self.data_file = os.path.expandvars(raw_path)
        self._rng = np.random.RandomState(seed)

        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        # deal with dataset
        # TODO: shuffule and iterate the dataset
        data_size = sum(1 for _ in open(self.data_file, "r"))
        self.task_idxs = range(data_size)

        # -------------------------- Ray actors setup --------------------------
        env_worker = ray.remote(**resources_per_worker)(WebVoyagerWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(seed + (i // self.group_n), self.data_file, self._env_kwargs)
            self._workers.append(worker)


    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list):
        # TODO: take care of the action formulation, action_key, and info (depends on action key)
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        idx = self._rng.choice(self.task_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()

        # Send reset commands to all workers
        futures = []
        for worker, i in zip(self._workers, idx):
            future = worker.reset.remote(i)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            future = self._workers[env_idx].render.remote(mode)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)
        
        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webvoyager_envs(
    config_path: str,
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    """Mirror *build_webvoyager_envs* so higher‑level code can swap seamlessly."""
    return WebVoyagerMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        config_path=config_path,
        env_kwargs=env_kwargs,
    )