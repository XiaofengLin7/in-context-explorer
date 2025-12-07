import types
from typing import Dict, List

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from verl.protocol import DataProto


class DummyTokenizer:
    pad_token_id = 0

    def batch_decode(self, responses, skip_special_tokens=True):
        batch_size = responses.shape[0]
        return ["look"] * batch_size


class DummyActorWorkerGroup:
    world_size = 1

    def generate_sequences(self, batch_input: DataProto) -> DataProto:
        batch_size = len(batch_input.batch["input_ids"])
        tensors = {
            "responses": torch.ones((batch_size, 1), dtype=torch.long),
            "input_ids": batch_input.batch["input_ids"].clone(),
            "attention_mask": batch_input.batch["attention_mask"].clone(),
            "position_ids": batch_input.batch["position_ids"].clone(),
        }
        return DataProto.from_dict(tensors=tensors)


class DummyEnvManager:
    def __init__(self):
        self.step_calls = 0
        self.soft_reset_calls = 0

    def reset(self, kwargs=None):
        self.step_calls = 0
        obs = {"text": ["init"], "image": None, "anchor": ["init"]}
        infos = [{"won": False, "is_action_valid": True}]
        return obs, infos

    def step(self, text_actions: List[str]):
        self.step_calls += 1
        if self.step_calls == 1:
            obs = {"text": ["success"], "image": None, "anchor": ["success"]}
            rewards = np.array([1.0], dtype=np.float32)
            dones = np.array([True], dtype=bool)
            infos = [{"won": True, "is_action_valid": True}]
        else:
            obs = {"text": ["fail"], "image": None, "anchor": ["fail"]}
            rewards = np.array([0.0], dtype=np.float32)
            dones = np.array([True], dtype=bool)
            infos = [{"won": False, "is_action_valid": True}]
        return obs, rewards, dones, infos

    def soft_reset(self, indices: List[int], infos: List[Dict]):
        self.soft_reset_calls += 1
        assert indices == [0]
        obs_updates = {
            "text": {0: "after-reset"},
            "image": {},
            "anchor": {0: "after-reset"},
        }
        info_updates = {0: {"won": False, "is_action_valid": True}}
        return obs_updates, info_updates

    def success_evaluator(self, total_infos, **kwargs):
        last_won = float(total_infos[0][-1]["won"])
        return {"success_rate": np.array([last_won], dtype=np.float32)}


class DummyCollector(TrajectoryCollector):
    def preprocess_batch(self, gen_batch: DataProto, obs: Dict) -> DataProto:
        batch_clone = gen_batch.batch.clone()
        non_tensor_clone = {key: val.copy() for key, val in gen_batch.non_tensor_batch.items()}
        return DataProto(batch=batch_clone, non_tensor_batch=non_tensor_clone, meta_info=dict(gen_batch.meta_info))


def make_config():
    env_cfg = types.SimpleNamespace(
        max_steps=4,
        rollout=types.SimpleNamespace(n=0),
        env_name="alfworld/AlfredTWEnv",
        success_coef=30.0,
        multi_episode_rollout=types.SimpleNamespace(enable=True, reward_per_completion=2.0),
    )
    data_cfg = types.SimpleNamespace()
    algo_cfg = types.SimpleNamespace(filter_groups=types.SimpleNamespace(max_num_gen_batches=1))
    return types.SimpleNamespace(env=env_cfg, data=data_cfg, algorithm=algo_cfg)


def make_gen_batch(batch_size: int = 1) -> DataProto:
    tensors = {
        "input_ids": torch.zeros((batch_size, 1), dtype=torch.long),
        "attention_mask": torch.ones((batch_size, 1), dtype=torch.long),
        "position_ids": torch.zeros((batch_size, 1), dtype=torch.long),
    }
    tensor_dict = TensorDict(source=tensors, batch_size=(batch_size,))
    non_tensors = {
        "raw_prompt": np.array(["hello"], dtype=object),
        "raw_prompt_ids": np.array([[0, 1]], dtype=object),
        "data_source": np.array(["train"], dtype=object),
        "env_kwargs": np.array([None], dtype=object),
    }
    return DataProto(batch=tensor_dict, non_tensor_batch=non_tensors)


def test_multi_episode_reward_and_success_counter():
    collector = DummyCollector(make_config(), DummyTokenizer())
    env = DummyEnvManager()
    actor = DummyActorWorkerGroup()
    gen_batch = make_gen_batch()

    _, episode_rewards, episode_lengths, success, _, _ = collector.vanilla_multi_turn_loop(
        gen_batch=gen_batch,
        actor_rollout_wg=actor,
        envs=env,
    )

    assert env.soft_reset_calls == 1
    assert success["avg_successes_within_max_steps"][0] == pytest.approx(1.0)
    assert episode_rewards[0] == pytest.approx(2.0)
    assert episode_lengths[0] == pytest.approx(2.0)

