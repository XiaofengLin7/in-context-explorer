"""Tests for WebShop ORBIT-style chat prompting and multi-episode support."""

import types
from typing import Any, Dict, List

import pytest

from agent_system.environments.env_manager import WebshopEnvironmentManager


class _EnvNS(types.SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key) if hasattr(self, key) else default


def make_config(
    prompt_type: str = "chat",
    history_length: int = 5,
    multi_episode: bool = True,
    episode_max_steps: int = 15,
) -> Any:
    cfg = types.SimpleNamespace()
    env = _EnvNS()
    env.prompt_type = prompt_type
    env.history_length = history_length
    env.max_steps = 30
    env.env_name = "Webshop"
    env.webshop = types.SimpleNamespace(use_small=True, human_goals=False)
    if multi_episode:
        env.multi_episode_rollout = types.SimpleNamespace(
            enable=True,
            reward_per_completion=1.0,
            episode_max_steps=episode_max_steps,
        )
    cfg.env = env
    data = types.SimpleNamespace()
    data.train_batch_size = 1
    data.val_batch_size = 1
    cfg.data = data
    return cfg


_AVAIL_ACTIONS = {
    "has_search_bar": True,
    "clickables": ["Back to Search", "Next >", "B09K3GL12Z", "Buy Now"],
}


class _FakeWebshopEnvs:
    def __init__(self):
        self._obs0 = [
            "WebShop [SEP] Instruction: [SEP] Buy a red t-shirt [SEP] Search"
        ]
        self._obs1 = [
            "WebShop [SEP] Instruction: [SEP] Buy a red t-shirt [SEP] "
            "Back to Search [SEP] Page 1 [SEP] Red T-Shirt $15"
        ]
        self._current_goal_idxs = [42]

    def reset(self):
        infos = [{"available_actions": _AVAIL_ACTIONS, "won": False}]
        return list(self._obs0), infos

    def soft_reset(self, indices):
        obs_map, info_map = {}, {}
        for idx in indices:
            obs_map[idx] = self._obs0[0]
            info_map[idx] = {"available_actions": _AVAIL_ACTIONS, "won": False}
        return obs_map, info_map

    def step(self, actions: List[str]):
        obs = list(self._obs1)
        rewards = [0.0]
        dones = [False]
        infos = [{"available_actions": _AVAIL_ACTIONS, "won": False, "task_score": 0.0}]
        return obs, rewards, dones, infos


def _projection_fn(text_actions: List[str]):
    """Simple projection that extracts action tag content."""
    actions = []
    valids = []
    for t in text_actions:
        lower = t.lower()
        start = lower.find("<action>")
        end = lower.find("</action>")
        if start != -1 and end != -1:
            actions.append(lower[start + 8 : end].strip())
            valids.append(1)
        else:
            actions.append(t[-20:])
            valids.append(0)
    return actions, valids


def _extract_text(msg: Dict) -> str:
    """Extract text from a chat message's content blocks."""
    content = msg.get("content", [])
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if block.get("type") == "text"
        )
    return str(content)


# ---- Tests ---- #


class TestChatResetReturnsMessageList:
    def test_returns_list_of_message_lists(self):
        cfg = make_config()
        mgr = WebshopEnvironmentManager(_FakeWebshopEnvs(), _projection_fn, cfg)
        obs, infos = mgr.reset(kwargs={})

        messages = obs["text"]
        assert isinstance(messages, list)
        assert len(messages) == 1  # batch_size=1
        assert isinstance(messages[0], list)

    def test_first_message_is_system(self):
        cfg = make_config()
        mgr = WebshopEnvironmentManager(_FakeWebshopEnvs(), _projection_fn, cfg)
        obs, _ = mgr.reset(kwargs={})

        messages = obs["text"][0]
        assert messages[0]["role"] == "system"
        system_text = _extract_text(messages[0])
        assert "WebShop" in system_text
        assert "<think>" in system_text
        assert "<action>" in system_text

    def test_second_message_is_user_with_obs_and_actions(self):
        cfg = make_config()
        mgr = WebshopEnvironmentManager(_FakeWebshopEnvs(), _projection_fn, cfg)
        obs, _ = mgr.reset(kwargs={})

        messages = obs["text"][0]
        assert len(messages) == 2  # system + user
        assert messages[1]["role"] == "user"
        user_text = _extract_text(messages[1])
        assert "Buy a red t-shirt" in user_text
        assert "admissible actions" in user_text.lower()
        assert "search[<your query>]" in user_text

    def test_anchor_is_raw_string(self):
        cfg = make_config()
        mgr = WebshopEnvironmentManager(_FakeWebshopEnvs(), _projection_fn, cfg)
        obs, _ = mgr.reset(kwargs={})

        assert isinstance(obs["anchor"], list)
        assert isinstance(obs["anchor"][0], str)

    def test_system_prompt_includes_episode_cap(self):
        cfg = make_config(episode_max_steps=15)
        mgr = WebshopEnvironmentManager(_FakeWebshopEnvs(), _projection_fn, cfg)
        obs, _ = mgr.reset(kwargs={})

        system_text = _extract_text(obs["text"][0][0])
        assert "15" in system_text

    def test_single_episode_system_prompt(self):
        cfg = make_config(multi_episode=False)
        mgr = WebshopEnvironmentManager(_FakeWebshopEnvs(), _projection_fn, cfg)
        obs, _ = mgr.reset(kwargs={})

        system_text = _extract_text(obs["text"][0][0])
        assert "episode" not in system_text.lower()
        assert "WebShop" in system_text


class TestChatStepBuildsProperHistory:
    def test_step_adds_assistant_and_user(self):
        cfg = make_config()
        mgr = WebshopEnvironmentManager(_FakeWebshopEnvs(), _projection_fn, cfg)
        mgr.reset(kwargs={})

        response = ["<think>I should search for red t-shirt.</think><action>search[red t-shirt]</action>"]
        next_obs, _, _, _ = mgr.step(response)

        messages = next_obs["text"][0]
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user"]

    def test_assistant_message_contains_raw_response(self):
        cfg = make_config()
        mgr = WebshopEnvironmentManager(_FakeWebshopEnvs(), _projection_fn, cfg)
        mgr.reset(kwargs={})

        response = ["<think>I should search.</think><action>search[red t-shirt]</action>"]
        next_obs, _, _, _ = mgr.step(response)

        messages = next_obs["text"][0]
        assistant_text = _extract_text(messages[2])
        assert "<think>I should search.</think>" in assistant_text
        assert "<action>search[red t-shirt]</action>" in assistant_text

    def test_content_is_typed_blocks(self):
        cfg = make_config()
        mgr = WebshopEnvironmentManager(_FakeWebshopEnvs(), _projection_fn, cfg)
        mgr.reset(kwargs={})

        response = ["<think>ok</think><action>search[shirt]</action>"]
        next_obs, _, _, _ = mgr.step(response)

        for msg in next_obs["text"][0]:
            assert isinstance(msg["content"], list)
            assert msg["content"][0]["type"] == "text"


class TestAvailableActionsStripping:
    def test_old_user_messages_have_no_actions(self):
        cfg = make_config()
        mgr = WebshopEnvironmentManager(_FakeWebshopEnvs(), _projection_fn, cfg)
        mgr.reset(kwargs={})

        response = ["<think>ok</think><action>search[red t-shirt]</action>"]
        next_obs, _, _, _ = mgr.step(response)

        messages = next_obs["text"][0]
        # messages[1] is the first (now historical) user message
        old_user_text = _extract_text(messages[1])
        assert "admissible" not in old_user_text.lower()
        assert "search[<your query>]" not in old_user_text

    def test_old_user_messages_preserve_observation(self):
        cfg = make_config()
        mgr = WebshopEnvironmentManager(_FakeWebshopEnvs(), _projection_fn, cfg)
        mgr.reset(kwargs={})

        response = ["<think>ok</think><action>search[red t-shirt]</action>"]
        next_obs, _, _, _ = mgr.step(response)

        messages = next_obs["text"][0]
        old_user_text = _extract_text(messages[1])
        assert "observation" in old_user_text.lower()

    def test_current_user_message_has_actions(self):
        cfg = make_config()
        mgr = WebshopEnvironmentManager(_FakeWebshopEnvs(), _projection_fn, cfg)
        mgr.reset(kwargs={})

        response = ["<think>ok</think><action>search[red t-shirt]</action>"]
        next_obs, _, _, _ = mgr.step(response)

        messages = next_obs["text"][0]
        current_user_text = _extract_text(messages[-1])
        assert "admissible actions" in current_user_text.lower()

    def test_multiple_steps_only_last_has_actions(self):
        cfg = make_config()
        envs = _FakeWebshopEnvs()
        mgr = WebshopEnvironmentManager(envs, _projection_fn, cfg)
        mgr.reset(kwargs={})

        for _ in range(3):
            response = ["<think>ok</think><action>search[shirt]</action>"]
            next_obs, _, _, _ = mgr.step(response)

        messages = next_obs["text"][0]
        user_messages = [m for m in messages if m["role"] == "user"]
        for um in user_messages[:-1]:
            assert "admissible" not in _extract_text(um).lower()
        assert "admissible" in _extract_text(user_messages[-1]).lower()


class TestChatEpisodeMarkers:
    def test_soft_reset_adds_episode_marker(self):
        cfg = make_config(episode_max_steps=3)
        envs = _FakeWebshopEnvs()
        mgr = WebshopEnvironmentManager(envs, _projection_fn, cfg)
        _, infos = mgr.reset(kwargs={})

        response = ["<think>ok</think><action>search[shirt]</action>"]
        _, _, _, infos_step = mgr.step(response)
        mgr.episode_step_ids[0] = 3

        infos_step[0]["multi_episode_soft_reset_reason"] = "step_limit"
        obs_updates, _ = mgr.soft_reset([0], infos_step)

        messages = obs_updates["text"][0]
        all_text = " ".join(_extract_text(m) for m in messages)
        assert "[Episode 2]" in all_text

    def test_soft_reset_includes_prev_episode_result(self):
        cfg = make_config(episode_max_steps=3)
        envs = _FakeWebshopEnvs()
        mgr = WebshopEnvironmentManager(envs, _projection_fn, cfg)
        _, infos = mgr.reset(kwargs={})

        response = ["<think>ok</think><action>search[shirt]</action>"]
        _, _, _, infos_step = mgr.step(response)
        mgr.episode_step_ids[0] = 3

        infos_step[0]["multi_episode_soft_reset_reason"] = "step_limit"
        obs_updates, _ = mgr.soft_reset([0], infos_step)

        messages = obs_updates["text"][0]
        all_text = " ".join(_extract_text(m) for m in messages)
        assert "Previous episode result:" in all_text

    def test_episode_ids_increment(self):
        cfg = make_config(episode_max_steps=3)
        envs = _FakeWebshopEnvs()
        mgr = WebshopEnvironmentManager(envs, _projection_fn, cfg)
        mgr.reset(kwargs={})

        assert mgr.episode_ids[0] == 0

        response = ["<think>ok</think><action>search[shirt]</action>"]
        _, _, _, infos_step = mgr.step(response)
        mgr.episode_step_ids[0] = 3
        infos_step[0]["multi_episode_soft_reset_reason"] = "success"
        mgr.soft_reset([0], infos_step)

        assert mgr.episode_ids[0] == 1


class TestVanillaMultiEpisode:
    """Test that vanilla prompt_type still works with multi-episode enabled."""

    def test_vanilla_reset_returns_strings(self):
        cfg = make_config(prompt_type="vanilla")
        mgr = WebshopEnvironmentManager(_FakeWebshopEnvs(), _projection_fn, cfg)
        obs, _ = mgr.reset(kwargs={})

        assert isinstance(obs["text"], list)
        assert isinstance(obs["text"][0], str)

    def test_vanilla_multi_episode_init_template(self):
        cfg = make_config(prompt_type="vanilla")
        mgr = WebshopEnvironmentManager(_FakeWebshopEnvs(), _projection_fn, cfg)
        obs, _ = mgr.reset(kwargs={})

        text = obs["text"][0]
        assert "episode" in text.lower()
        assert "15" in text  # episode_cap
