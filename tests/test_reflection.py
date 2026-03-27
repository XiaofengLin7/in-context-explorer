"""Tests for ORBIT-style reflection mechanism in multi-episode chat prompting."""

import types
from typing import Any, Dict, List

import pytest

from agent_system.environments.env_manager import WebshopEnvironmentManager
from agent_system.environments.prompts.webshop import WEBSHOP_CHAT_REFLECTION_PROMPT
from agent_system.memory import AlfWorldChatMemory


class _EnvNS(types.SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key) if hasattr(self, key) else default


def make_config(
    prompt_type: str = "chat",
    history_length: int = 5,
    episode_max_steps: int = 15,
    enable_reflection: bool = True,
) -> Any:
    cfg = types.SimpleNamespace()
    env = _EnvNS()
    env.prompt_type = prompt_type
    env.history_length = history_length
    env.max_steps = 30
    env.env_name = "Webshop"
    env.webshop = types.SimpleNamespace(use_small=True, human_goals=False)
    env.multi_episode_rollout = types.SimpleNamespace(
        enable=True,
        reward_per_completion=1.0,
        episode_max_steps=episode_max_steps,
        enable_reflection=enable_reflection,
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
    content = msg.get("content", [])
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if block.get("type") == "text"
        )
    return str(content)


# ---- Memory tests ---- #


class TestReflectionInChatMemory:
    def test_reflection_record_renders_in_history(self):
        """Reflection prompt and response appear correctly in message history."""
        from agent_system.environments.prompts.webshop import WEBSHOP_CHAT_USER_OBS_STRIPPED

        mem = AlfWorldChatMemory(stripped_template=WEBSHOP_CHAT_USER_OBS_STRIPPED)
        mem.reset(batch_size=1)

        # Record 0: initial obs
        mem.store({
            'user_text': ["Your task is: Buy a red t-shirt. Obs: Search page"],
            'assistant_text': [None],
            'raw_obs': ["Search page"],
            'admissible_actions': ["search[query]"],
            'episode_id': [0],
            'episode_step': [0],
            'episode_label': [""],
        })

        # Record 1: normal step
        mem.store({
            'user_text': ["Obs: Results page. Actions: click[item1]"],
            'assistant_text': ["<think>search</think><action>search[red t-shirt]</action>"],
            'raw_obs': ["Results page"],
            'admissible_actions': ["click[item1]"],
            'episode_id': [0],
            'episode_step': [1],
            'episode_label': [""],
        })

        # Record 2: reflection record
        mem.store_single(0, {
            'user_text': "Terminal obs. Episode result: did not succeed.\n\n[Reflection] ...",
            'assistant_text': "I should have searched differently. <remark>Try broader search.</remark>",
            'raw_obs': "Terminal obs",
            'admissible_actions': '',
            'episode_id': 0,
            'episode_step': 1,
            'episode_label': '',
            'is_reflection': True,
        })

        # Record 3: new episode init
        mem.store_single(0, {
            'user_text': "Your task is: Buy a red t-shirt. Obs: Search page. Actions: search[query]",
            'assistant_text': None,
            'raw_obs': "Search page",
            'admissible_actions': "search[query]",
            'episode_id': 1,
            'episode_step': 0,
            'episode_label': '',
        })

        messages = mem.build_message_history(["System prompt"])[0]
        roles = [m["role"] for m in messages]

        # Expected: system, user, assistant, user, assistant, user, user
        # (init, response1, obs1_stripped, reflection_response, reflection_stripped, new_ep_obs)
        assert roles[0] == "system"
        # Verify reflection response appears as assistant
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert any("<remark>" in _extract_text(m) for m in assistant_msgs)

    def test_reflection_stripped_in_history(self):
        """Historical reflection user messages use abbreviated format."""
        from agent_system.environments.prompts.webshop import WEBSHOP_CHAT_USER_OBS_STRIPPED

        mem = AlfWorldChatMemory(stripped_template=WEBSHOP_CHAT_USER_OBS_STRIPPED)
        mem.reset(batch_size=1)

        # Initial record
        mem.store({
            'user_text': ["Initial obs"],
            'assistant_text': [None],
            'raw_obs': ["Initial obs"],
            'admissible_actions': [""],
            'episode_id': [0],
            'episode_step': [0],
            'episode_label': [""],
        })

        # Reflection record
        mem.store_single(0, {
            'user_text': "Terminal obs.\n\n[Reflection] full prompt here...",
            'assistant_text': "My reflection text",
            'raw_obs': "Terminal obs",
            'admissible_actions': '',
            'episode_id': 0,
            'episode_step': 1,
            'episode_label': 'previous episode 1 failed',
            'is_reflection': True,
        })

        # New episode initial (this becomes the "last" record)
        mem.store_single(0, {
            'user_text': "New episode obs with actions",
            'assistant_text': None,
            'raw_obs': "New episode obs",
            'admissible_actions': "search[query]",
            'episode_id': 1,
            'episode_step': 0,
            'episode_label': '',
        })

        messages = mem.build_message_history(["System"])[0]
        user_msgs = [m for m in messages if m["role"] == "user"]

        # The reflection user message (not last) should be stripped
        # It should contain "[Reflection requested" marker
        reflection_user = user_msgs[1]  # second user msg = reflection (stripped)
        reflection_text = _extract_text(reflection_user)
        assert "[Reflection requested" in reflection_text
        assert "Terminal obs" in reflection_text
        # Should NOT contain the full reflection prompt
        assert "[Reflection] full prompt" not in reflection_text


# ---- env_manager tests ---- #


class TestWebshopReflectionEnvManager:
    def test_enable_reflection_from_config(self):
        cfg = make_config(enable_reflection=True)
        mgr = WebshopEnvironmentManager(_FakeWebshopEnvs(), _projection_fn, cfg)
        assert mgr.enable_reflection is True

    def test_reflection_disabled_by_default(self):
        cfg = make_config(enable_reflection=False)
        mgr = WebshopEnvironmentManager(_FakeWebshopEnvs(), _projection_fn, cfg)
        assert mgr.enable_reflection is False

    def test_build_reflection_obs_appends_prompt(self):
        cfg = make_config(enable_reflection=True)
        envs = _FakeWebshopEnvs()
        mgr = WebshopEnvironmentManager(envs, _projection_fn, cfg)
        obs, infos = mgr.reset(kwargs={})

        # Do one step
        response = ["<think>ok</think><action>search[red t-shirt]</action>"]
        next_obs, _, _, step_infos = mgr.step(response)

        # Trigger reflection
        prev_infos = [{"won": False, "multi_episode_soft_reset_reason": "terminal"}]
        mgr.build_reflection_obs([0], next_obs, prev_infos)

        # Check that the reflection prompt is in the last user message
        messages = next_obs["text"][0]
        last_user = [m for m in messages if m["role"] == "user"][-1]
        last_user_text = _extract_text(last_user)
        assert "[Reflection]" in last_user_text
        assert "reflect on the past experience" in last_user_text.lower()

    def test_consume_reflection_stores_and_resets(self):
        cfg = make_config(enable_reflection=True, episode_max_steps=3)
        envs = _FakeWebshopEnvs()
        mgr = WebshopEnvironmentManager(envs, _projection_fn, cfg)
        obs, infos = mgr.reset(kwargs={})

        # Do one step
        response = ["<think>ok</think><action>search[red t-shirt]</action>"]
        next_obs, _, _, step_infos = mgr.step(response)

        # Trigger reflection
        prev_info = {
            "won": False,
            "multi_episode_soft_reset_reason": "step_limit",
            "multi_episode_episode_step_count": 3,
            "multi_episode_episode_success": 0.0,
        }
        mgr.build_reflection_obs([0], next_obs, [prev_info])

        # Verify reflection record is in memory
        last_rec = mgr.memory[0][-1]
        assert last_rec.get("is_reflection") is True
        assert last_rec["assistant_text"] is None

        # Consume reflection
        reflection_text = ["I should have searched for a different keyword. <remark>Try broader terms.</remark>"]
        obs_updates, info_updates = mgr.consume_reflection([0], reflection_text, [prev_info])

        # Verify reflection text was stored
        reflection_rec = None
        for rec in mgr.memory[0]:
            if rec.get("is_reflection"):
                reflection_rec = rec
                break
        assert reflection_rec is not None
        assert "<remark>" in reflection_rec["assistant_text"]

        # Verify soft_reset happened (episode_id incremented)
        assert mgr.episode_ids[0] == 1
        assert mgr.episode_step_ids[0] == 0

        # Verify new observation returned
        assert 0 in obs_updates["text"]

    def test_reflection_appears_in_subsequent_history(self):
        """After reflection + soft_reset, the reflection text is in the chat history."""
        cfg = make_config(enable_reflection=True, episode_max_steps=3)
        envs = _FakeWebshopEnvs()
        mgr = WebshopEnvironmentManager(envs, _projection_fn, cfg)
        obs, infos = mgr.reset(kwargs={})

        # Step
        response = ["<think>ok</think><action>search[red t-shirt]</action>"]
        next_obs, _, _, step_infos = mgr.step(response)

        # Trigger and consume reflection
        prev_info = {
            "won": False,
            "multi_episode_soft_reset_reason": "step_limit",
            "multi_episode_episode_step_count": 3,
            "multi_episode_episode_success": 0.0,
        }
        mgr.build_reflection_obs([0], next_obs, [prev_info])
        reflection_text = ["My reflection with <remark>key insight</remark>"]
        obs_updates, _ = mgr.consume_reflection([0], reflection_text, [prev_info])

        # The new observation should contain the reflection text in history
        messages = obs_updates["text"][0]
        all_text = " ".join(_extract_text(m) for m in messages)
        assert "[Reflection requested" in all_text  # stripped reflection user msg
        # The reflection response should be in an assistant message
        assistant_texts = [_extract_text(m) for m in messages if m["role"] == "assistant"]
        assert any("<remark>key insight</remark>" in t for t in assistant_texts)

    def test_episode_boundary_after_reflection(self):
        """After reflection, the new episode has proper [Episode N] boundary marker."""
        cfg = make_config(enable_reflection=True, episode_max_steps=3)
        envs = _FakeWebshopEnvs()
        mgr = WebshopEnvironmentManager(envs, _projection_fn, cfg)
        obs, infos = mgr.reset(kwargs={})

        # Step to make episode non-trivial
        response = ["<think>ok</think><action>search[red t-shirt]</action>"]
        next_obs, _, _, step_infos = mgr.step(response)
        mgr.episode_step_ids[0] = 3

        # Trigger and consume reflection
        prev_info = {
            "won": False,
            "multi_episode_soft_reset_reason": "step_limit",
            "multi_episode_episode_step_count": 3,
            "multi_episode_episode_success": 0.0,
        }
        mgr.build_reflection_obs([0], next_obs, [prev_info])
        obs_updates, _ = mgr.consume_reflection(
            [0], ["Reflection text <remark>plan</remark>"], [prev_info])

        messages = obs_updates["text"][0]
        all_text = " ".join(_extract_text(m) for m in messages)
        assert "[Episode 2]" in all_text
