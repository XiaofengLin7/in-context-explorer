import types
from typing import Any, Dict, List, Tuple

import pytest

from agent_system.environments.env_manager import (
    extract_known_and_unknown,
    select_prompt_variant,
    WebshopEnvironmentManager,
    AlfWorldEnvironmentManager,
)


class _EnvNS(types.SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key) if hasattr(self, key) else default


def make_config(prompt_type: str = "summary", history_length: int = 1) -> Any:
    cfg = types.SimpleNamespace()
    env = _EnvNS()
    env.prompt_type = prompt_type
    env.history_length = history_length
    # Minimal required fields for env-specific code paths
    env.env_name = "test"
    cfg.env = env
    # Minimal required for data access in some constructors
    data = types.SimpleNamespace()
    data.train_batch_size = 1
    data.val_batch_size = 1
    cfg.data = data
    return cfg


def test_extract_known_unknown_with_tags():
    responses = [
        "<think><known>Known A</known><unknown>Unknown A</unknown></think><action>do[x]</action>",
        "<think>prefix <known>Known B</known> mid <unknown>Unknown B</unknown> suffix</think>",
    ]
    known, unknown = extract_known_and_unknown(responses)
    assert known == ["Known A", "Known B"]
    assert unknown == ["Unknown A", "Unknown B"]


def test_extract_known_unknown_with_headers():
    responses = [
        "<think>Known: K1\nSome text\nUnknown: U1</think>",
        "<think>KNOWN INFORMATION: K2\n---\nUNKNOWN: U2</think>",
    ]
    known, unknown = extract_known_and_unknown(responses)
    assert known == ["K1\nSome text", "K2\n---"]
    assert unknown == ["U1", "U2"]


def test_select_prompt_variant_returns_expected_flags():
    cfg_summary = make_config(prompt_type="summary")
    init_s, hist_s, keep_s = select_prompt_variant(
        cfg_summary, "v_init", "v_hist", "s_init", "s_hist"
    )
    assert (init_s, hist_s, keep_s) == ("s_init", "s_hist", True)

    cfg_vanilla = make_config(prompt_type="vanilla")
    init_v, hist_v, keep_v = select_prompt_variant(
        cfg_vanilla, "v_init", "v_hist", "s_init", "s_hist"
    )
    assert (init_v, hist_v, keep_v) == ("v_init", "v_hist", False)


class _FakeWebshopEnvs:
    def __init__(self):
        self._obs0 = "CTX [SEP] Instruction: [SEP] Find a red mug [SEP] Page: home"
        self._obs1 = "CTX [SEP] Instruction: [SEP] Find a red mug [SEP] Page: results"

    def reset(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        infos = [
            {
                "available_actions": {
                    "has_search_bar": True,
                    "clickables": ["Deals", "Mugs"],
                }
            }
        ]
        return [self._obs0], infos

    def step(self, actions: List[str]):
        next_obs = [self._obs1]
        rewards = [0.0]
        dones = [False]
        infos = [
            {
                "available_actions": {
                    "has_search_bar": True,
                    "clickables": ["Red Mug", "Blue Mug"],
                }
            }
        ]
        return next_obs, rewards, dones, infos


def test_webshop_summary_pipeline_injects_known_unknown():
    cfg = make_config(prompt_type="summary", history_length=1)

    def projection_f(text_actions: List[str]):
        return ["search[red mug]"], [True]

    mgr = WebshopEnvironmentManager(_FakeWebshopEnvs(), projection_f, cfg)
    observations, infos = mgr.reset(kwargs={})
    assert "WebShop" in observations["text"][0]

    agent_resp = [
        "<think><known>We are on the home page.</known><unknown>The best brand.</unknown></think>"
        "<action>search[red mug]</action>"
    ]
    next_obs, rewards, dones, infos = mgr.step(agent_resp)
    t = next_obs["text"][0]
    assert "<known>We are on the home page.</known>" in t
    assert "<unknown>The best brand.</unknown>" in t


class _FakeAlfEnvs:
    def __init__(self):
        self._text0 = ["You are in the kitchen. You see a fridge. Your task is to: open the fridge."]
        self._text1 = ["You opened the fridge. You see an apple."]
        # In this codebase, admissible commands are treated as a list, not a method
        self.get_admissible_commands = [["OpenFridge", "Look"]]

    def reset(self):
        image_obs = None
        infos = [{"extra.gamefile": None}]
        return self._text0, image_obs, infos

    def step(self, actions: List[str]):
        text_obs = self._text1
        image_obs = None
        rewards = [0.0]
        dones = [False]
        infos = [{"extra.gamefile": None}]
        return text_obs, image_obs, rewards, dones, infos


def test_alfworld_summary_pipeline_injects_known_unknown():
    cfg = make_config(prompt_type="summary", history_length=1)

    def projection_f(text_actions: List[str], admissible):
        return ["OpenFridge"], [True]

    mgr = AlfWorldEnvironmentManager(_FakeAlfEnvs(), projection_f, cfg)
    observations, infos = mgr.reset(kwargs={})
    assert "ALFRED" in observations["text"][0]

    agent_resp = [
        "<think><known>Fridge is closed.</known><unknown>Where the apple is.</unknown></think>"
        "<action>OpenFridge</action>"
    ]
    next_obs, rewards, dones, infos = mgr.step(agent_resp)
    t = next_obs["text"][0]
    assert "<known>Fridge is closed.</known>" in t
    assert "<unknown>Where the apple is.</unknown>" in t


