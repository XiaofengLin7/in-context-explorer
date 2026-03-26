"""Tests for ALFWorld ORBIT-style chat prompting (prompt_type='chat')."""

import types
from typing import Any, Dict, List, Tuple

import pytest

from agent_system.environments.env_manager import AlfWorldEnvironmentManager


class _EnvNS(types.SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key) if hasattr(self, key) else default


def make_config(
    prompt_type: str = "chat",
    history_length: int = 5,
    multi_episode: bool = True,
    episode_max_steps: int = 10,
) -> Any:
    cfg = types.SimpleNamespace()
    env = _EnvNS()
    env.prompt_type = prompt_type
    env.history_length = history_length
    env.max_steps = 30
    env.env_name = "alfworld/AlfredTWEnv"
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


class _FakeAlfEnvs:
    def __init__(self):
        self._text0 = [
            "You are in the kitchen. You see a fridge. Your task is to: open the fridge."
        ]
        self._text1 = ["You opened the fridge. You see an apple."]
        self.get_admissible_commands = [
            ["open fridge 1", "go to counter 1", "go to shelf 1", "look", "help"]
        ]

    def reset(self):
        image_obs = None
        infos = [{"extra.gamefile": "fake_game"}]
        return self._text0, image_obs, infos

    def soft_reset(self, env_indices: List[int], gamefiles: List[str]):
        text_obs_map = {idx: self._text0[0] for idx in env_indices}
        image_obs_map: Dict[int, None] = {}
        info_map = {
            idx: {"extra.gamefile": gamefiles[i]}
            for i, idx in enumerate(env_indices)
        }
        return text_obs_map, image_obs_map, info_map

    def step(self, actions: List[str]):
        text_obs = self._text1
        image_obs = None
        rewards = [0.0]
        dones = [False]
        infos = [{"extra.gamefile": None}]
        return text_obs, image_obs, rewards, dones, infos


def _projection_fn(text_actions: List[str], admissible):
    """Simple projection that extracts action tag content or passes through."""
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
            actions.append(t[-30:])
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
        mgr = AlfWorldEnvironmentManager(_FakeAlfEnvs(), _projection_fn, cfg)
        obs, infos = mgr.reset(kwargs={})

        messages = obs["text"]
        assert isinstance(messages, list)
        assert len(messages) == 1  # batch_size=1
        assert isinstance(messages[0], list)

    def test_first_message_is_system(self):
        cfg = make_config()
        mgr = AlfWorldEnvironmentManager(_FakeAlfEnvs(), _projection_fn, cfg)
        obs, _ = mgr.reset(kwargs={})

        messages = obs["text"][0]
        assert messages[0]["role"] == "system"
        system_text = _extract_text(messages[0])
        assert "ALFRED" in system_text
        assert "<think>" in system_text
        assert "<action>" in system_text

    def test_second_message_is_user_with_obs_and_actions(self):
        cfg = make_config()
        mgr = AlfWorldEnvironmentManager(_FakeAlfEnvs(), _projection_fn, cfg)
        obs, _ = mgr.reset(kwargs={})

        messages = obs["text"][0]
        assert len(messages) == 2  # system + user
        assert messages[1]["role"] == "user"
        user_text = _extract_text(messages[1])
        assert "kitchen" in user_text
        assert "admissible actions" in user_text
        assert "open fridge 1" in user_text

    def test_anchor_is_raw_string(self):
        cfg = make_config()
        mgr = AlfWorldEnvironmentManager(_FakeAlfEnvs(), _projection_fn, cfg)
        obs, _ = mgr.reset(kwargs={})

        assert isinstance(obs["anchor"], list)
        assert isinstance(obs["anchor"][0], str)
        assert "kitchen" in obs["anchor"][0]

    def test_system_prompt_includes_episode_cap(self):
        cfg = make_config(episode_max_steps=10)
        mgr = AlfWorldEnvironmentManager(_FakeAlfEnvs(), _projection_fn, cfg)
        obs, _ = mgr.reset(kwargs={})

        system_text = _extract_text(obs["text"][0][0])
        assert "10" in system_text

    def test_single_episode_system_prompt(self):
        cfg = make_config(multi_episode=False)
        mgr = AlfWorldEnvironmentManager(_FakeAlfEnvs(), _projection_fn, cfg)
        obs, _ = mgr.reset(kwargs={})

        system_text = _extract_text(obs["text"][0][0])
        assert "episode" not in system_text.lower()
        assert "ALFRED" in system_text


class TestChatStepBuildsProperHistory:
    def test_step_adds_assistant_and_user(self):
        cfg = make_config()
        mgr = AlfWorldEnvironmentManager(_FakeAlfEnvs(), _projection_fn, cfg)
        mgr.reset(kwargs={})

        response = ["<think>I should open the fridge.</think><action>open fridge 1</action>"]
        next_obs, _, _, _ = mgr.step(response)

        messages = next_obs["text"][0]
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user"]

    def test_assistant_message_contains_raw_response(self):
        cfg = make_config()
        mgr = AlfWorldEnvironmentManager(_FakeAlfEnvs(), _projection_fn, cfg)
        mgr.reset(kwargs={})

        response = ["<think>I should open the fridge.</think><action>open fridge 1</action>"]
        next_obs, _, _, _ = mgr.step(response)

        messages = next_obs["text"][0]
        assistant_text = _extract_text(messages[2])
        assert "<think>I should open the fridge.</think>" in assistant_text
        assert "<action>open fridge 1</action>" in assistant_text

    def test_content_is_typed_blocks(self):
        """Verify content uses [{"type": "text", "text": "..."}] format."""
        cfg = make_config()
        mgr = AlfWorldEnvironmentManager(_FakeAlfEnvs(), _projection_fn, cfg)
        mgr.reset(kwargs={})

        response = ["<think>ok</think><action>look</action>"]
        next_obs, _, _, _ = mgr.step(response)

        for msg in next_obs["text"][0]:
            assert isinstance(msg["content"], list)
            assert msg["content"][0]["type"] == "text"


class TestAdmissibleActionsStripping:
    def test_old_user_messages_have_no_admissible_actions(self):
        cfg = make_config()
        mgr = AlfWorldEnvironmentManager(_FakeAlfEnvs(), _projection_fn, cfg)
        mgr.reset(kwargs={})

        response = ["<think>ok</think><action>open fridge 1</action>"]
        next_obs, _, _, _ = mgr.step(response)

        messages = next_obs["text"][0]
        # messages[1] is the first (now historical) user message
        old_user_text = _extract_text(messages[1])
        assert "admissible" not in old_user_text.lower()

    def test_old_user_messages_preserve_observation(self):
        cfg = make_config()
        mgr = AlfWorldEnvironmentManager(_FakeAlfEnvs(), _projection_fn, cfg)
        mgr.reset(kwargs={})

        response = ["<think>ok</think><action>open fridge 1</action>"]
        next_obs, _, _, _ = mgr.step(response)

        messages = next_obs["text"][0]
        old_user_text = _extract_text(messages[1])
        assert "kitchen" in old_user_text

    def test_current_user_message_has_admissible_actions(self):
        cfg = make_config()
        mgr = AlfWorldEnvironmentManager(_FakeAlfEnvs(), _projection_fn, cfg)
        mgr.reset(kwargs={})

        response = ["<think>ok</think><action>open fridge 1</action>"]
        next_obs, _, _, _ = mgr.step(response)

        messages = next_obs["text"][0]
        # Last message is the current user message
        current_user_text = _extract_text(messages[-1])
        assert "admissible actions" in current_user_text.lower()

    def test_multiple_steps_only_last_has_actions(self):
        cfg = make_config()
        envs = _FakeAlfEnvs()
        mgr = AlfWorldEnvironmentManager(envs, _projection_fn, cfg)
        mgr.reset(kwargs={})

        for _ in range(3):
            response = ["<think>ok</think><action>look</action>"]
            next_obs, _, _, _ = mgr.step(response)

        messages = next_obs["text"][0]
        user_messages = [m for m in messages if m["role"] == "user"]
        # All but last should be stripped
        for um in user_messages[:-1]:
            assert "admissible" not in _extract_text(um).lower()
        # Last should have them
        assert "admissible" in _extract_text(user_messages[-1]).lower()


class TestChatEpisodeMarkers:
    def test_soft_reset_adds_episode_marker(self):
        cfg = make_config(episode_max_steps=3)
        envs = _FakeAlfEnvs()
        mgr = AlfWorldEnvironmentManager(envs, _projection_fn, cfg)
        _, infos = mgr.reset(kwargs={})

        # Simulate a step
        response = ["<think>ok</think><action>look</action>"]
        _, _, _, infos_step = mgr.step(response)
        mgr.episode_step_ids[0] = 3

        # Soft reset
        infos_step[0]["multi_episode_soft_reset_reason"] = "step_limit"
        obs_updates, _ = mgr.soft_reset([0], infos_step)

        messages = obs_updates["text"][0]
        all_text = " ".join(_extract_text(m) for m in messages)
        assert "[Episode 2]" in all_text

    def test_soft_reset_includes_prev_episode_result(self):
        cfg = make_config(episode_max_steps=3)
        envs = _FakeAlfEnvs()
        mgr = AlfWorldEnvironmentManager(envs, _projection_fn, cfg)
        _, infos = mgr.reset(kwargs={})

        # Simulate a step
        response = ["<think>ok</think><action>look</action>"]
        _, _, _, infos_step = mgr.step(response)
        mgr.episode_step_ids[0] = 3

        # Soft reset
        infos_step[0]["multi_episode_soft_reset_reason"] = "step_limit"
        obs_updates, _ = mgr.soft_reset([0], infos_step)

        messages = obs_updates["text"][0]
        all_text = " ".join(_extract_text(m) for m in messages)
        assert "Previous episode result:" in all_text

    def test_help_excluded_from_admissible(self):
        cfg = make_config()
        mgr = AlfWorldEnvironmentManager(_FakeAlfEnvs(), _projection_fn, cfg)
        obs, _ = mgr.reset(kwargs={})

        current_user_text = _extract_text(obs["text"][0][-1])
        assert "help" not in current_user_text.split("admissible")[1].lower().split("'")
