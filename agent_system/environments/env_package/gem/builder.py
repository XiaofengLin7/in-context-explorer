from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gem
import numpy as np
import ray

# ------------------- Default GEM task pools ------------------- #
# Default task pools (fixed env_id + seed) for reproducible sampling.
#
# We intentionally include MANY fixed seeds per game to reduce variance and
# avoid overfitting to a single instance.
GEM_TRAIN_ENV_IDS: List[str] = [
    "game:GuessTheNumber-v0-easy",
    "game:Mastermind-v0-easy",
    "game:Minesweeper-v0-easy",
    "game:Sudoku-v0-easy",
    "game:Hangman-v0-easy",
    "game:TowerofHanoi-v0-easy",
]

GEM_EVAL_ENV_IDS: List[str] = [
    "game:Wordle-v0-easy",
    "game:FifteenPuzzle-v0-easy",
]

DEFAULT_GEM_TRAIN_NUM_SEEDS = 100
DEFAULT_GEM_EVAL_NUM_SEEDS = 100


def _build_task_pool(env_ids: Sequence[str], num_seeds: int, seed_base: int) -> List[Dict[str, Any]]:
    """
    Build a deterministic (env_id, seed) task pool.

    We offset each env's seed range to avoid accidental collisions across games.
    """
    pool: List[Dict[str, Any]] = []
    stride = 100_000
    for env_idx, env_id in enumerate(env_ids):
        base = seed_base + env_idx * stride
        for k in range(int(num_seeds)):
            pool.append({"env_id": env_id, "seed": int(base + k)})
    return pool


GEM_TASK_POOL_TRAIN: List[Dict[str, Any]] = _build_task_pool(
    env_ids=GEM_TRAIN_ENV_IDS,
    num_seeds=DEFAULT_GEM_TRAIN_NUM_SEEDS,
    seed_base=0,
)

GEM_TASK_POOL_EVAL: List[Dict[str, Any]] = _build_task_pool(
    env_ids=GEM_EVAL_ENV_IDS,
    num_seeds=DEFAULT_GEM_EVAL_NUM_SEEDS,
    seed_base=1_000_000,
)

class GemWorker:
    """
    Ray remote actor holding a single GEM environment instance.
    Supports task pools with fixed env_id/seed for reproducible resets.
    """

    def __init__(
        self,
        env_ids: Sequence[str],
        seed: int,
        max_steps: Optional[int] = None,
        task_pool: Optional[Sequence[Dict[str, Any]]] = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.env_ids: List[str] = list(env_ids)
        self.max_steps = max_steps
        self._steps = 0
        self.task_pool = list(task_pool) if task_pool is not None else None
        self.current_task: Optional[Dict[str, Any]] = None

        env_id, task_seed = self._sample_task()
        self.env_id = env_id
        self.env = self._make_env(env_id)
        if task_seed is not None:
            self.env.reset(seed=task_seed)

    def _sample_env_id(self) -> str:
        return self.rng.choice(self.env_ids).item()

    def _sample_task(self) -> Tuple[str, Optional[int]]:
        """
        Sample a task from the task_pool if provided; otherwise sample env_id only.
        Each task dict may include {'env_id': str, 'seed': Optional[int], ...}.
        """
        if self.task_pool:
            task = self.rng.choice(self.task_pool)
            env_id = task.get("env_id", self._sample_env_id())
            task_seed = task.get("seed")
            self.current_task = {"env_id": env_id, "seed": task_seed, **task}
            return env_id, task_seed
        env_id = self._sample_env_id()
        self.current_task = {"env_id": env_id, "seed": None}
        return env_id, None

    def reset(self, override_env_ids: Optional[Sequence[str]] = None):
        if override_env_ids is None:
            env_id, task_seed = self._sample_task()
        else:
            env_id = self.rng.choice(override_env_ids).item()
            task_seed = None
            self.current_task = {"env_id": env_id, "seed": None}

        self.env_id = env_id
        self.env = self._make_env(env_id)
        self._steps = 0
        obs, info = self.env.reset(seed=task_seed)
        info = info or {}
        info["env_id"] = env_id
        return obs, info

    def soft_reset(self):
        """
        Reset to the same task instance (env_id + seed) if available.
        Falls back to a normal reset if no current task is tracked.
        """
        if self.current_task is None:
            return self.reset()
        env_id = self.current_task.get("env_id")
        task_seed = self.current_task.get("seed")
        self.env_id = env_id
        self.env = self._make_env(env_id)
        self._steps = 0
        obs, info = self.env.reset(seed=task_seed)
        info = info or {}
        info["env_id"] = env_id
        return obs, info

    def _make_env(self, env_id: str):
        """
        Create a GEM env instance.

        Some GEM games (Wordle/Hangman) call `nltk.download("words")` during __init__,
        which can race when many Ray actors initialize simultaneously. We pre-ensure
        the corpus exists using a simple file lock.
        """
        if env_id.lower().startswith("game:wordle") or env_id.lower().startswith("game:hangman"):
            _ensure_nltk_words()
            # GEM's Wordle/Hangman envs still call nltk.download("words") (non-quiet) in __init__,
            # which spams logs across Ray workers. Temporarily force quiet=True.
            try:
                import nltk  # type: ignore

                original_download = nltk.download

                def _quiet_download(*args: Any, **kwargs: Any):
                    kwargs.setdefault("quiet", True)
                    return original_download(*args, **kwargs)

                nltk.download = _quiet_download  # type: ignore[assignment]
                return gem.make(env_id)
            finally:
                try:
                    nltk.download = original_download  # type: ignore[assignment]
                except Exception:
                    pass
        return gem.make(env_id)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        done = bool(terminated or truncated)
        if self.max_steps is not None and self._steps >= self.max_steps:
            done = True
        info = info or {}
        info.setdefault("env_id", self.env_id)
        # Emit a unified success signal for multi-episode rollout logic.
        # GEM game envs consistently use "Congratulations!" for success terminal messages.
        won = bool(terminated) and str(obs).lstrip().startswith("Congratulations!")
        info["won"] = won
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)
        # Detect "internal max turns" termination from GEM envs.
        # GEM game envs consistently emit "You have reached the maximum number of turns" on this condition.
        obs_text = str(obs)
        info["internal_max_turns"] = bool(truncated) and ("maximum number of turns" in obs_text.lower())
        return obs, float(reward), done, info


def _ensure_nltk_words() -> None:
    """
    Ensure NLTK 'words' corpus exists, avoiding concurrent download races.
    """
    # Prefer user-provided NLTK_DATA; otherwise default to a per-user cache dir.
    # This avoids hard-coding any lab-specific paths and works across machines.
    default_dir = str(Path.home() / ".cache" / "nltk_data")
    download_dir = os.environ.get("NLTK_DATA", default_dir)
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    try:
        import nltk  # type: ignore

        if download_dir not in nltk.data.path:
            nltk.data.path.insert(0, download_dir)
        try:
            nltk.data.find("corpora/words")
            return
        except LookupError:
            pass

        lock_path = Path(download_dir) / ".nltk_words.lock"
        acquired = False
        while not acquired:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                acquired = True
            except FileExistsError:
                time.sleep(0.2)

        try:
            # Re-check under lock.
            try:
                nltk.data.find("corpora/words")
                return
            except LookupError:
                nltk.download("words", download_dir=download_dir, quiet=True)
        finally:
            try:
                lock_path.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception:
        # If nltk is not available or download fails, let GEM env raise its own error.
        return


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
        task_pool: Optional[Sequence[Dict[str, Any]]],
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
            worker = env_worker.remote(self.env_ids, worker_seed, max_steps, task_pool)
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

    def soft_reset(self, indices: Sequence[int]):
        """
        Reset specific workers to their current task instance (env_id + seed).
        """
        futures = []
        for idx in indices:
            futures.append((idx, self.workers[idx].soft_reset.remote()))

        obs_map: Dict[int, Any] = {}
        info_map: Dict[int, Dict[str, Any]] = {}
        results = ray.get([f for _, f in futures])
        for (idx, _), (obs, info) in zip(futures, results):
            obs_map[idx] = obs
            info_map[idx] = info
        return obs_map, info_map

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
    task_pool: Optional[Sequence[Dict[str, Any]]] = None,
    use_default_pool: bool = False,
    is_train: bool = True,
) -> GemMultiProcessEnv:
    """
    Public factory following the other env builders.

    If task_pool is None and use_default_pool is True:
      - uses GEM_TASK_POOL_TRAIN when is_train=True
      - uses GEM_TASK_POOL_EVAL when is_train=False
    """
    resources = resources_per_worker or {"num_cpus": 0.1}
    resolved_task_pool = task_pool
    if resolved_task_pool is None and use_default_pool:
        resolved_task_pool = GEM_TASK_POOL_TRAIN if is_train else GEM_TASK_POOL_EVAL

    return GemMultiProcessEnv(
        env_ids=env_ids,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources,
        max_steps=max_steps,
        task_pool=resolved_task_pool,
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

    GEM games generally parse the *last* \\boxed{...} in the model output.
    To reduce format errors, we:
      - extract the last \\boxed{...} if present and send only that
      - validate the extracted content with a per-game regex
    """

    def _extract_last_boxed(text: str) -> Optional[str]:
        matches = list(re.finditer(r"\\boxed\{([^}]*)\}", text, flags=re.DOTALL))
        if not matches:
            return None
        return matches[-1].group(1).strip()

    def _wrap_boxed(content: str) -> str:
        return f"\\boxed{{{content.strip()}}}"

    def _is_valid_for_env(env_id: str, boxed_content: str) -> bool:
        eid = env_id.lower()
        c = boxed_content.strip()
        if eid.startswith("game:guessthenumber"):
            return re.fullmatch(r"[+-]?\d+", c) is not None
        if eid.startswith("game:mastermind"):
            # e.g. "1 2" or "1 2 3 4"
            return re.fullmatch(r"(?:\d+\s+)*\d+", c) is not None
        if eid.startswith("game:minesweeper"):
            return re.fullmatch(r"(reveal|flag)\s+\d+\s+\d+", c, flags=re.IGNORECASE) is not None
        if eid.startswith("game:sudoku"):
            return re.fullmatch(r"\d+\s+\d+\s+\d+", c) is not None
        if eid.startswith("game:hangman"):
            return re.fullmatch(r"[a-zA-Z]+(?:\s+[a-zA-Z]+)*", c) is not None
        if eid.startswith("game:towerofhanoi"):
            return re.fullmatch(r"[ABCabc]\s*,?\s*[ABCabc]", c) is not None
        if eid.startswith("game:wordle"):
            return re.fullmatch(r"[a-zA-Z]+(?:\s+[a-zA-Z]+)*", c) is not None
        if eid.startswith("game:fifteenpuzzle"):
            return re.fullmatch(r"(up|down|left|right)", c, flags=re.IGNORECASE) is not None
        # Fallback: if it is boxed, accept it.
        return True

    actions: List[Any] = []
    valids: List[bool] = []
    for idx, act in enumerate(text_actions):
        env_id = env_ids[idx] if env_ids is not None and idx < len(env_ids) else ""
        boxed = _extract_last_boxed(act)
        if boxed is None:
            # Enforce protocol strictly: action must contain exactly one \\boxed{...}.
            action = act
            valid = False
        else:
            action = _wrap_boxed(boxed)
            valid = _is_valid_for_env(env_id, boxed)
        actions.append(action)
        valids.append(valid)
    return actions, valids
