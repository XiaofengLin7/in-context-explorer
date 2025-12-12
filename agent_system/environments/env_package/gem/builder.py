from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gem
import numpy as np
import ray


class GemWorker:
    """
    Ray remote actor holding a single GEM environment instance.
    """

    def __init__(self, env_ids: Sequence[str], seed: int, max_steps: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.env_ids: List[str] = list(env_ids)
        self.max_steps = max_steps
        self._steps = 0
        env_id = self._sample_env_id()
        self.env_id = env_id
        self.env = gem.make(env_id)

    def _sample_env_id(self) -> str:
        return self.rng.choice(self.env_ids).item()

    def reset(self, override_env_ids: Optional[Sequence[str]] = None):
        env_id = self._sample_env_id() if override_env_ids is None else self.rng.choice(override_env_ids).item()
        self.env_id = env_id
        self.env = gem.make(env_id)
        self._steps = 0
        obs, info = self.env.reset()
        info = info or {}
        info["env_id"] = env_id
        return obs, info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        done = bool(terminated or truncated)
        if self.max_steps is not None and self._steps >= self.max_steps:
            done = True
        info = info or {}
        info.setdefault("env_id", self.env_id)
        if done:
            obs, info = self.reset()
        return obs, float(reward), done, info


class GemMultiProcessEnv:
    """
    Ray-based parallel environment wrapper for GEM environments.
    Mirrors patterns used in alfworld/gym_cards wrappers.
    """

    def __init__(
        self,
        env_ids: Sequence[str],
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: Dict[str, Any],
        max_steps: Optional[int],
        is_train: bool = True,
    ) -> None:
        if not ray.is_initialized():
            ray.init()

        self.env_ids: List[str] = list(env_ids)
        self.env_num = env_num
        self.group_n = group_n if group_n > 0 else 1
        self.num_processes = self.env_num * self.group_n
        self.is_train = is_train

        env_worker = ray.remote(**resources_per_worker)(GemWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker_seed = seed + (i // self.group_n)
            worker = env_worker.remote(self.env_ids, worker_seed, max_steps)
            self.workers.append(worker)

    def step(self, actions: Sequence[Any]):
        assert len(actions) == self.num_processes, "actions length must equal num_processes"
        futures = [w.step.remote(a) for w, a in zip(self.workers, actions)]
        results = ray.get(futures)

        obs_list: List[Any] = []
        rewards_list: List[float] = []
        dones_list: List[bool] = []
        info_list: List[Dict[str, Any]] = []

        for obs, reward, done, info in results:
            obs_list.append(obs)
            rewards_list.append(reward)
            dones_list.append(done)
            info_list.append(info)

        if isinstance(obs_list[0], np.ndarray):
            obs_list = np.array(obs_list)

        return obs_list, rewards_list, dones_list, info_list

    def reset(self, override_env_ids: Optional[Sequence[str]] = None):
        futures = [w.reset.remote(override_env_ids) for w in self.workers]
        results = ray.get(futures)

        obs_list: List[Any] = []
        info_list: List[Dict[str, Any]] = []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        if isinstance(obs_list[0], np.ndarray):
            obs_list = np.array(obs_list)

        return obs_list, info_list

    def close(self):
        if ray.is_initialized():
            for worker in self.workers:
                try:
                    ray.kill(worker)
                except Exception:
                    pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def build_gem_envs(
    env_ids: Sequence[str],
    seed: int,
    env_num: int,
    group_n: int,
    max_steps: Optional[int],
    resources_per_worker: Optional[Dict[str, Any]] = None,
    is_train: bool = True,
) -> GemMultiProcessEnv:
    """
    Public factory following the other env builders.
    """
    resources = resources_per_worker or {"num_cpus": 0.1}
    return GemMultiProcessEnv(
        env_ids=env_ids,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources,
        max_steps=max_steps,
        is_train=is_train,
    )


def _parse_boxed_number(text: str) -> Optional[int]:
    """
    Extract an integer inside \\boxed{...}. Returns None on failure.
    """
    match = re.search(r"\\boxed\{([^}]*)\}", text)
    if not match:
        return None
    candidate = match.group(1).strip()
    # Remove any trailing punctuation
    candidate = re.sub(r"[^\d\-+]+$", "", candidate)
    try:
        return int(candidate)
    except ValueError:
        return None


def gem_projection(
    text_actions: Sequence[str], env_ids: Optional[Sequence[str]] = None
) -> Tuple[List[Any], List[bool]]:
    """
    Projection function to map model text to GEM actions.

    - If env_id suggests a numeric guess game, try to parse \\boxed{number}.
    - Otherwise, pass through the raw text.
    """
    actions: List[Any] = []
    valids: List[bool] = []
    for idx, act in enumerate(text_actions):
        env_id = env_ids[idx] if env_ids is not None and idx < len(env_ids) else None
        action = act
        valid = True

        # Heuristic: numeric guessing games start with "game:GuessTheNumber"
        if env_id and env_id.lower().startswith("game:guessthenumber"):
            parsed = _parse_boxed_number(act)
            if parsed is not None:
                action = act  # keep original string; GEM expects string input
            else:
                valid = False
        actions.append(action)
        valids.append(valid)
    return actions, valids
