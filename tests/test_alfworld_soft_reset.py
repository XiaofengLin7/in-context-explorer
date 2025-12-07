import types
from typing import Any, Dict, List, Tuple

import pytest

from agent_system.environments.env_manager import AlfWorldEnvironmentManager


def make_config(prompt_type: str = "vanilla", history_length: int = 1) -> Any:
    cfg = types.SimpleNamespace()
    env = types.SimpleNamespace()
    env.prompt_type = prompt_type
    env.history_length = history_length
    env.env_name = "alfworld/AlfredTWEnv"
    cfg.env = env
    data = types.SimpleNamespace()
    data.train_batch_size = 1
    data.val_batch_size = 1
    cfg.data = data
    return cfg


class _StubAlfworldEnvs:
    def __init__(self):
        self.get_admissible_commands = [["look", "open"]]
        self.soft_reset_args: Tuple[List[int], Tuple[str, ...]] | None = None

    def reset(self, kwargs=None):
        text_obs = [
            "You are in the kitchen. Your task is to: open the fridge."
        ]
        image_obs = None
        infos = [
            {
                "extra.gamefile": "game_A",
                "admissible_commands": self.get_admissible_commands[0],
            }
        ]
        return text_obs, image_obs, infos

    def soft_reset(self, indices: List[int], gamefiles: List[str]):
        self.soft_reset_args = (indices, tuple(gamefiles))
        idx = indices[0]
        gamefile = gamefiles[0]
        text_obs_map = {
            idx: f"You are still in the kitchen. Task file: {gamefile}"
        }
        image_obs_map: Dict[int, None] = {}
        info_map = {
            idx: {
                "extra.gamefile": gamefile,
                "admissible_commands": self.get_admissible_commands[0],
            }
        }
        return text_obs_map, image_obs_map, info_map


def _projection_fn(text_actions: List[str], admissible) -> Tuple[List[str], List[bool]]:
    return text_actions, [True for _ in text_actions]


def test_soft_reset_reuses_original_gamefile():
    cfg = make_config()
    envs = _StubAlfworldEnvs()
    manager = AlfWorldEnvironmentManager(envs, _projection_fn, cfg)

    observations, infos = manager.reset(kwargs={})
    assert "open the fridge" in observations["text"][0]
    assert infos[0]["extra.gamefile"] == "game_A"

    obs_updates, info_updates = manager.soft_reset([0], infos)

    assert envs.soft_reset_args == ([0], ("game_A",))
    assert "Task file: game_A" in obs_updates["text"][0]
    assert obs_updates["anchor"][0] == "You are still in the kitchen. Task file: game_A"
    assert info_updates[0]["extra.gamefile"] == "game_A"

