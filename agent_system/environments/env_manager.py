# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory, WebVoyagerMemory
from omegaconf import OmegaConf
import re
def _get_openable_class_set():
    from agent_system.environments.env_package.alfworld.alfworld.gen.constants import OPENABLE_CLASS_SET
    return OPENABLE_CLASS_SET

def extract_known_and_unknown(responses: List[str]) -> Tuple[List[str], List[str]]:
    """
    Extract known and unknown information from text actions.

    Searches for <known>...</known> and <unknown>...</unknown> inside <think>...</think>.
    Falls back to parsing "Known:" / "Unknown:" headers inside <think> when tags are missing.
    """
    known_information: List[str] = []
    unknown_information: List[str] = []

    def _extract_by_headers(inside: str) -> Tuple[str, str]:
        # Normalize variants like "Known Information -:" to "Known:"
        norm = re.sub(r"(?i)\b(known|unknown)\s*(information)?\s*[-–—]?\s*:", r"\1:", inside)
        flags = re.IGNORECASE | re.DOTALL | re.MULTILINE
        known_m = re.search(r"^\s*Known\s*:\s*(.*?)(?=^\s*Unknown\s*:|\Z)", norm, flags)
        unknown_m = re.search(r"^\s*Unknown\s*:\s*(.*?)(?=^\s*[A-Z][^\n]*:|\Z)", norm, flags)
        known = known_m.group(1).strip() if known_m else ""
        unknown = unknown_m.group(1).strip() if unknown_m else ""
        return known, unknown

    for response in responses:
        # Extract content within <think>...</think>
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
        if not think_match:
            known_information.append("")
            unknown_information.append("")
            continue
        think_content = think_match.group(1)

        # Prefer explicit tags
        known_match = re.search(r"<known>(.*?)</known>", think_content, re.DOTALL | re.IGNORECASE)
        unknown_match = re.search(r"<unknown>(.*?)</unknown>", think_content, re.DOTALL | re.IGNORECASE)

        known = known_match.group(1).strip() if known_match else ""
        unknown = unknown_match.group(1).strip() if unknown_match else ""

        # Fallback to header-style sections inside <think>
        if not known and not unknown:
            known, unknown = _extract_by_headers(think_content)

        known_information.append(known)
        unknown_information.append(unknown)

    return known_information, unknown_information

def select_prompt_variant(
    config,
    vanilla_init: str,
    vanilla_history: str,
    summary_init: str,
    summary_history: str,
    gold_init: str | None = None,
    gold_history: str | None = None,
) -> Tuple[str, str, bool]:
    """
    Return (prompt_init, prompt_history, keep_known_and_unknown) based on config.env.prompt_type.
    """
    prompt_type = config.env.get('prompt_type', 'vanilla')
    if prompt_type == 'summary':
        return summary_init, summary_history, True
    if prompt_type == 'vanilla':
        return vanilla_init, vanilla_history, False
    if prompt_type == 'gold':
        if gold_init is None or gold_history is None:
            # Fall back to vanilla variant if gold templates are not provided.
            return vanilla_init, vanilla_history, False
        return gold_init, gold_history, False
    if prompt_type == 'chat':
        # Chat mode is handled separately by AlfWorldEnvironmentManager;
        # fall back to vanilla for other envs that call this function.
        return vanilla_init, vanilla_history, False
    raise ValueError(f"Invalid prompt type: {config.env.prompt_type}")

ALFWORLD_TASK_TYPES = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]

def extract_task_type(gamefile: str) -> str | None:
    """Extract ALFWorld task type from a gamefile string."""
    if not gamefile:
        return None
    for task in ALFWORLD_TASK_TYPES:
        if task in gamefile:
            return task
    return None

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos




class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        prompt_type = getattr(config.env, 'prompt_type', 'vanilla')
        if prompt_type == 'chat':
            from agent_system.memory import AlfWorldChatMemory
            self.memory = AlfWorldChatMemory()
        else:
            self.memory = SimpleMemory()
        multi_episode_cfg = getattr(config.env, "multi_episode_rollout", None)
        self.multi_episode_enabled = bool(getattr(multi_episode_cfg, "enable", False)) if multi_episode_cfg else False
        self.episode_max_steps = getattr(multi_episode_cfg, "episode_max_steps", None) if multi_episode_cfg else None
        self.enable_reflection = bool(getattr(multi_episode_cfg, "enable_reflection", False)) if multi_episode_cfg else False
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.episode_ids = [0 for _ in range(len(text_obs))]
        self.episode_step_ids = [0 for _ in range(len(text_obs))]
        self.episode_labels = [""] * len(text_obs)
        self.prev_episode_labels = [""] * len(text_obs)
        self.episode_step_ids = [0 for _ in range(len(text_obs))]
        self._reflection_pending = [False] * len(text_obs)
        self.tasks = []
        self.visited_receptacles = [set() for _ in range(len(text_obs))]
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)
        self.receptacles = self.extract_receptacles(self.envs.get_admissible_commands)
        # Initialize unvisited_receptacles as a copy of all receptacles
        self.unvisited_receptacles = [receptacle_set.copy() for receptacle_set in self.receptacles]
        if self.config.env.prompt_type == 'chat':
            full_text_obs = self._build_chat_obs(text_obs, self.envs.get_admissible_commands, init=True)
        elif self.config.env.prompt_type == 'gold':
            full_text_obs = self.build_text_obs_gold(text_obs, self.envs.get_admissible_commands, init=True)
        elif self.config.env.prompt_type == 'summary':
            full_text_obs = self.build_text_obs_with_known_and_unknown(text_obs, self.envs.get_admissible_commands, [], [],init=True)
        else:
            full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def soft_reset(self, env_indices: List[int], prev_infos: List[Dict[str, Any]]):
        if not env_indices:
            return {}, {}

        env_indices = [int(idx) for idx in env_indices]
        gamefiles = []
        for idx in env_indices:
            info = prev_infos[idx]
            gamefile = info.get("extra.gamefile") or self.gamefile[idx]
            if gamefile is None:
                raise ValueError(f"Environment index {idx} has no associated gamefile for soft reset.")
            gamefiles.append(gamefile)

        text_obs_map, image_obs_map, info_map = self.envs.soft_reset(env_indices, gamefiles)

        # Update cached raw observations for later memory fetches.
        for idx in env_indices:
            raw_text = text_obs_map[idx]
            self.pre_text_obs[idx] = raw_text
            self.gamefile[idx] = info_map[idx].get("extra.gamefile", gamefiles[env_indices.index(idx)])
            last_episode_steps = self.episode_step_ids[idx]
            if last_episode_steps == 0 and self.memory and len(self.memory[idx]) > 0:
                last_episode_steps = int(self.memory[idx][-1].get("episode_step", 0))
            reason = prev_infos[idx].get("multi_episode_soft_reset_reason", "success")
            prev_episode = self.episode_ids[idx] + 1
            self.episode_ids[idx] += 1
            self.episode_step_ids[idx] = 0
            if self.multi_episode_enabled:
                current_episode = self.episode_ids[idx] + 1
                episode_cap = int(self.episode_max_steps or self.config.env.max_steps)
                if reason == "success":
                    label = f"previous episode {prev_episode} succeeded in {last_episode_steps} step(s)"
                else:
                    label = (
                        f"previous episode {prev_episode} reached {last_episode_steps}/{episode_cap} step(s) "
                        f"without success"
                    )
                # Tag the last record of the previous episode so history shows its outcome.
                if self.memory and len(self.memory[idx]) > 0:
                    self.memory[idx][-1]["episode_label"] = label
                self.prev_episode_labels[idx] = label
                self.episode_labels[idx] = ""
            else:
                self.episode_labels[idx] = ""
                self.prev_episode_labels[idx] = ""

        if self.config.env.prompt_type == 'chat':
            # Store new episode init observation for each reset env
            for idx in env_indices:
                reformatted = "\n ".join(
                    f"'{s}'" for s in self.envs.get_admissible_commands[idx] if s != 'help'
                )
                user_text = ALFWORLD_CHAT_USER_OBS.format(
                    current_observation=self.pre_text_obs[idx],
                    admissible_actions=reformatted,
                )
                self.memory.store_single(idx, {
                    'user_text': user_text,
                    'assistant_text': None,
                    'raw_obs': self.pre_text_obs[idx],
                    'admissible_actions': reformatted,
                    'episode_id': self.episode_ids[idx],
                    'episode_step': self.episode_step_ids[idx],
                    'episode_label': self.episode_labels[idx],
                })
            system_prompts = self._build_chat_system_prompts(len(self.pre_text_obs))
            full_text_obs = self.memory.build_message_history(system_prompts)
        elif self.config.env.prompt_type == 'gold':
            full_text_obs = self.build_text_obs_gold(self.pre_text_obs, self.envs.get_admissible_commands)
        elif self.config.env.prompt_type == 'summary':
            empty_known = [""] * len(self.pre_text_obs)
            empty_unknown = [""] * len(self.pre_text_obs)
            full_text_obs = self.build_text_obs_with_known_and_unknown(
                self.pre_text_obs,
                self.envs.get_admissible_commands,
                empty_known,
                empty_unknown,
            )
        else:
            full_text_obs = self.build_text_obs(self.pre_text_obs, self.envs.get_admissible_commands)

        obs_updates = {"text": {}, "image": {}, "anchor": {}}
        for idx in env_indices:
            obs_updates["text"][idx] = full_text_obs[idx]
            obs_updates["anchor"][idx] = self.pre_text_obs[idx]
            obs_updates["image"][idx] = image_obs_map.get(idx)

        return obs_updates, info_map
    
    def step(self, text_actions: List[str]):
        # extract known and unkown here as text_actions will be mutated in place in  self.projection_f
        if self.config.env.prompt_type == 'summary':
            known_information, unknown_information = extract_known_and_unknown(text_actions)
        # Save raw model responses before projection mutates text_actions in place
        if self.config.env.prompt_type == 'chat':
            raw_responses = list(text_actions)
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        for idx in range(len(actions)):
            if not self._reflection_pending[idx]:
                self.episode_step_ids[idx] += 1

        if self.config.env.prompt_type == 'chat':
            # Skip chat record storage for reflection-pending envs
            # (their step results will be overridden by consume_reflection)
            if any(self._reflection_pending):
                self._store_chat_record_partial(text_obs, raw_responses)
            else:
                self._store_chat_record(text_obs, raw_responses)
            full_text_obs = self._build_chat_obs(text_obs, self.envs.get_admissible_commands)
        else:
            self.memory.store({
                'text_obs': self.pre_text_obs,
                'action': actions,
                'episode_id': list(self.episode_ids),
                'episode_step': list(self.episode_step_ids),
                'episode_label': list(self.episode_labels),
            })

        self.pre_text_obs = text_obs
        self.update_receptacles(text_obs, actions)
        if self.config.env.prompt_type == 'chat':
            pass  # already built above
        elif self.config.env.prompt_type == 'summary':
            full_text_obs = self.build_text_obs_with_known_and_unknown(text_obs, self.envs.get_admissible_commands, known_information, unknown_information)
        elif self.config.env.prompt_type == 'gold':
            full_text_obs = self.build_text_obs_gold(text_obs, self.envs.get_admissible_commands)
        else:
            full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)

        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def normalize_receptacle(self, receptacle: str) -> str:
        """
        Normalize receptacle name to format: receptacle_class + one space + num_id.
        
        Input can have receptacle_class + multiple spaces (0 to many) + num_id.
        This function normalizes it to exactly one space between class and ID.
        
        Args:
            receptacle: Receptacle name (e.g., "dresser  1", "shelf5", "drawer   1")
            
        Returns:
            Normalized receptacle name with exactly one space (e.g., "dresser 1", "shelf 5", "drawer 1")
        """
        # Match letters (receptacle class) + zero or more spaces + digits (num_id)
        # Replace with receptacle_class + one space + num_id
        normalized = re.sub(r'([a-zA-Z]+)\s*(\d+)', r'\1 \2', receptacle)
        return normalized
    
    def is_openable_receptacle(self, receptacle: str) -> bool:
        for openable_receptacle in _get_openable_class_set():
            if openable_receptacle.lower() in receptacle.lower():
                return True
        return False

    def update_receptacles(self, text_obs: List[str], actions: List[str]):
        for i, action in enumerate(actions):
            if "go to" in action:
                parts = action.split("go to ", 1)
                if len(parts) < 2:
                    continue
                receptacle = parts[1]
                receptacle = self.normalize_receptacle(receptacle)
                if self.is_openable_receptacle(receptacle):
                    continue
                if receptacle in text_obs[i] and receptacle in self.receptacles[i]:
                    self.visited_receptacles[i].add(receptacle)
                    self.unvisited_receptacles[i].discard(receptacle)
            elif "open" in action:
                parts = action.split("open ", 1)
                if len(parts) < 2:
                    continue
                receptacle = parts[1]
                receptacle = self.normalize_receptacle(receptacle)
                if self.is_openable_receptacle(receptacle) and "is open" in text_obs[i] and receptacle in self.receptacles[i]:
                    self.visited_receptacles[i].add(receptacle)
                    self.unvisited_receptacles[i].discard(receptacle)
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")

    def extract_receptacles(self, admissible_actions: List[List[str]])->List[List[str]]:
        receptacles = []
        for actions in admissible_actions:
            current_receptacle = set()
            for action in actions:
                if "go to" in action:
                    current_receptacle.add(action.split("go to ")[1])
            receptacles.append(current_receptacle)
        return receptacles
        
    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        memory_contexts = valid_lens = None
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action",
                    episode_key="episode_id" if self.multi_episode_enabled else None,
                    episode_step_key="episode_step" if self.multi_episode_enabled else None,
                    episode_label_key="episode_label" if self.multi_episode_enabled else None)
        episode_cap = int(self.episode_max_steps or self.config.env.max_steps) if self.multi_episode_enabled else None

        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if self.multi_episode_enabled:
                previous_label = self.prev_episode_labels[i]
                if not previous_label and self.memory and len(self.memory[i]) > 0:
                    previous_label = self.memory[i][-1].get("episode_label", "")
                if init or self.config.env.history_length <= 0:
                    obs = ALFWORLD_TEMPLATE_MULTI_EPISODE_INIT.format(
                        episode_cap=episode_cap,
                        current_observation=text_obs[i],
                        admissible_actions=reformatted_admissible_actions,
                    )
                else:
                    history_text = memory_contexts[i] if memory_contexts is not None else ""
                    current_ep = self.episode_ids[i] + 1
                    if self.episode_step_ids[i] == 0:
                        if not history_text:
                            # No history yet for the new episode: include previous result if available.
                            if previous_label:
                                history_text = (
                                    f"--- Previous episode result: {previous_label} ---\n"
                                    f"--- Episode {current_ep} start ---"
                                )
                            else:
                                history_text = f"--- Episode {current_ep} start ---"
                        elif f"Episode {current_ep} start" not in history_text:
                            history_text = f"{history_text}\n--- Episode {current_ep} start ---"
                    obs = ALFWORLD_TEMPLATE_MULTI_EPISODE.format(
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=history_text,
                        current_episode=self.episode_ids[i] + 1,
                        current_step=self.episode_step_ids[i] + 1,
                        current_observation=text_obs[i],
                        admissible_actions=reformatted_admissible_actions,
                        episode_cap=episode_cap,
                    )
            else:
                if init or self.config.env.history_length <= 0:
                    obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                        current_observation=text_obs[i],
                        admissible_actions=reformatted_admissible_actions
                    )
                else:
                    history_text = memory_contexts[i] if memory_contexts is not None else ""
                    obs = ALFWORLD_TEMPLATE.format(
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=history_text,
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                        admissible_actions=reformatted_admissible_actions
                    )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs
    
    def build_text_obs_with_known_and_unknown(self, text_obs: List[str], admissible_actions: List[List[str]], known_information: List[str], unknown_information: List[str], init: bool = False) -> List[str]:
        postprocess_text_obs = []
        memory_contexts = valid_lens = None
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action",
                    episode_key="episode_id" if self.multi_episode_enabled else None,
                    episode_step_key="episode_step" if self.multi_episode_enabled else None,
                    episode_label_key="episode_label" if self.multi_episode_enabled else None)

        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init:
                obs = ALFWORLD_TEMPLATE_INIT_SUMMARY.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            elif self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS_SUMMARY.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    current_step=len(self.memory[i]) + 1,
                    admissible_actions=reformatted_admissible_actions,
                    known_information=known_information[i],
                    unknown_information=unknown_information[i]
                )
            else:
                history_text = memory_contexts[i] if memory_contexts is not None else ""
                obs = ALFWORLD_TEMPLATE_SUMMARY.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=history_text,
                    known_information=known_information[i],
                    unknown_information=unknown_information[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            obs = self._prepend_episode_message(i, obs)
            postprocess_text_obs.append(obs)
        return postprocess_text_obs
    
    def build_text_obs_gold(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        postprocess_text_obs = []
        memory_contexts = valid_lens = None
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action",
                    episode_key="episode_id" if self.multi_episode_enabled else None,
                    episode_step_key="episode_step" if self.multi_episode_enabled else None,
                    episode_label_key="episode_label" if self.multi_episode_enabled else None)

        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            visited_receptacles = ", ".join(sorted(list(self.visited_receptacles[i])))
            unvisited_receptacles = ", ".join(sorted(list(self.unvisited_receptacles[i])))

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                history_text = memory_contexts[i] if memory_contexts is not None else ""
                obs = ALFWORLD_TEMPLATE_GOLD.format(
                    task_description=self.tasks[i],
                    admissible_actions=reformatted_admissible_actions,
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=history_text,
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    visited_receptacles=visited_receptacles,
                    unvisited_receptacles=unvisited_receptacles
                )

            obs = self._prepend_episode_message(i, obs)
            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _prepend_episode_message(self, idx: int, prompt: str) -> str:
        if not self.multi_episode_enabled:
            return prompt
        message = self.episode_start_messages[idx]
        if message:
            prompt = f"{message}\n\n{prompt}"
            self.episode_start_messages[idx] = ""
        return prompt

    # ---- Chat (ORBIT-style multi-turn) helpers ---- #

    def _build_chat_system_prompts(self, batch_size: int) -> List[str]:
        if self.multi_episode_enabled:
            episode_cap = int(self.episode_max_steps or self.config.env.max_steps)
            return [ALFWORLD_CHAT_SYSTEM_PROMPT.format(episode_cap=episode_cap)] * batch_size
        return [ALFWORLD_CHAT_SYSTEM_PROMPT_SINGLE] * batch_size

    def _format_admissible_actions(self, admissible_commands: List[str]) -> str:
        return "\n ".join(f"'{s}'" for s in admissible_commands if s != 'help')

    def _build_chat_obs(
        self,
        text_obs: List[str],
        admissible_commands: List[List[str]],
        init: bool = False,
        raw_responses: List[str] = None,
    ) -> List[List[Dict]]:
        """Build chat message lists for all environments.

        On init: stores initial record (assistant_text=None) and builds messages.
        On step: assumes _store_chat_record was already called.
        """
        if init:
            user_texts = []
            raw_obs_list = []
            admissible_strs = []
            for i in range(len(text_obs)):
                reformatted = self._format_admissible_actions(admissible_commands[i])
                user_text = ALFWORLD_CHAT_USER_OBS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted,
                )
                user_texts.append(user_text)
                raw_obs_list.append(text_obs[i])
                admissible_strs.append(reformatted)

            self.memory.store({
                'user_text': user_texts,
                'assistant_text': [None] * len(text_obs),
                'raw_obs': raw_obs_list,
                'admissible_actions': admissible_strs,
                'episode_id': list(self.episode_ids),
                'episode_step': list(self.episode_step_ids),
                'episode_label': list(self.episode_labels),
            })

        system_prompts = self._build_chat_system_prompts(len(text_obs))
        return self.memory.build_message_history(system_prompts)

    def _store_chat_record(self, text_obs: List[str], raw_responses: List[str]):
        """Store a chat record pairing raw model responses with new observations."""
        user_texts = []
        admissible_strs = []
        for i in range(len(text_obs)):
            reformatted = self._format_admissible_actions(self.envs.get_admissible_commands[i])
            user_text = ALFWORLD_CHAT_USER_OBS.format(
                current_observation=text_obs[i],
                admissible_actions=reformatted,
            )
            user_texts.append(user_text)
            admissible_strs.append(reformatted)

        self.memory.store({
            'user_text': user_texts,
            'assistant_text': raw_responses,
            'raw_obs': list(text_obs),
            'admissible_actions': admissible_strs,
            'episode_id': list(self.episode_ids),
            'episode_step': list(self.episode_step_ids),
            'episode_label': list(self.episode_labels),
        })

    def _store_chat_record_partial(self, text_obs: List[str], raw_responses: List[str]):
        """Store chat records only for envs NOT in reflection-pending state."""
        for i in range(len(text_obs)):
            if self._reflection_pending[i]:
                continue
            reformatted = self._format_admissible_actions(self.envs.get_admissible_commands[i])
            user_text = ALFWORLD_CHAT_USER_OBS.format(
                current_observation=text_obs[i],
                admissible_actions=reformatted,
            )
            self.memory.store_single(i, {
                'user_text': user_text,
                'assistant_text': raw_responses[i],
                'raw_obs': text_obs[i],
                'admissible_actions': reformatted,
                'episode_id': self.episode_ids[i],
                'episode_step': self.episode_step_ids[i],
                'episode_label': self.episode_labels[i],
            })

    def build_reflection_obs(self, env_indices: List[int], next_obs: Dict, prev_infos: List[Dict]):
        """Append reflection prompt to terminal observation for completed episodes.

        Stores a reflection record in memory and rebuilds the message history
        so the model sees the reflection prompt as the next user message.
        """
        for idx in env_indices:
            self._reflection_pending[idx] = True
            terminal_obs = self.pre_text_obs[idx]
            episode_result = "succeeded" if prev_infos[idx].get("won", False) else "did not succeed"

            user_text = (
                f"Your current observation is: {terminal_obs}\n"
                f"Episode result: {episode_result}.\n\n"
                f"{ALFWORLD_CHAT_REFLECTION_PROMPT}"
            )

            self.memory.store_single(idx, {
                'user_text': user_text,
                'assistant_text': None,
                'raw_obs': terminal_obs,
                'admissible_actions': '',
                'episode_id': self.episode_ids[idx],
                'episode_step': self.episode_step_ids[idx],
                'episode_label': '',
                'is_reflection': True,
            })

        system_prompts = self._build_chat_system_prompts(len(self.pre_text_obs))
        full_text_obs = self.memory.build_message_history(system_prompts)

        for idx in env_indices:
            next_obs["text"][idx] = full_text_obs[idx]

    def consume_reflection(self, env_indices: List[int], text_actions: List[str], prev_infos: List[Dict]):
        """Store reflection responses in memory and soft_reset to start new episodes.

        Args:
            env_indices: Which environments are completing reflection.
            text_actions: The full text_actions list from the rollout step (indexed by env).
            prev_infos: Per-index saved infos from when the episode originally completed.
        """
        for idx in env_indices:
            last_rec = self.memory[idx][-1]
            assert last_rec.get('is_reflection', False), \
                f"Expected reflection record for env {idx}, got {last_rec.keys()}"
            last_rec['assistant_text'] = text_actions[idx]
            self._reflection_pending[idx] = False

        # Build prev_infos list indexed by env_idx for soft_reset
        infos_for_reset = [None] * len(self.pre_text_obs)
        for idx, info in zip(env_indices, prev_infos):
            infos_for_reset[idx] = info

        return self.soft_reset(env_indices, infos_for_reset)

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)

                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        task = extract_task_type(gamefile)
        if task:
            success.setdefault(f"{task}_success_rate", []).append(won_value)


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class GemEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for GEM text-based games.
    Uses SimpleMemory to keep a short history; prompts are the raw observations plus history.
    """

    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
        self.last_env_ids: List[str] = []
        self.current_obs: List[str] = []
        self.current_suffix: List[str] = []

        multi_episode_cfg = getattr(config.env, "multi_episode_rollout", None)
        self.multi_episode_enabled = bool(getattr(multi_episode_cfg, "enable", False)) if multi_episode_cfg else False
        self.episode_max_steps = getattr(multi_episode_cfg, "episode_max_steps", None) if multi_episode_cfg else None
        self.episode_ids: List[int] = []
        self.episode_step_ids: List[int] = []
        self.episode_labels: List[str] = []

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset()
        batch_size = len(obs)
        self.memory.reset(batch_size=batch_size)
        self.last_env_ids = [info.get("env_id", "") for info in infos]
        self.current_obs = list(obs)
        self.current_suffix = [str(info.get("suffix", "") or "") for info in infos]
        self.episode_ids = [0 for _ in range(batch_size)]
        self.episode_step_ids = [0 for _ in range(batch_size)]
        self.episode_labels = ["" for _ in range(batch_size)]
        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy(),
        }
        return observations, infos

    def step(self, text_actions: List[str]):
        # Project using env_ids to allow action parsing per game
        actions, valids = self.projection_f(text_actions, self.last_env_ids)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        # increment episode step counters
        for idx in range(len(actions)):
            self.episode_step_ids[idx] += 1

        # store history
        # If the action is invalid, store a short placeholder instead of the raw model output.
        # This keeps history compact and avoids polluting it with long reasoning text.
        stored_actions: List[Any] = []
        for act, valid in zip(actions, valids):
            if bool(valid):
                stored_actions.append(act)
            else:
                stored_actions.append("Invalid action (ignored). Please respond with a single valid \\boxed{...} action.")
        self.memory.store({
            "text_obs": self.current_obs,
            "action": stored_actions,
            "episode_id": list(self.episode_ids),
            "episode_step": list(self.episode_step_ids),
            "episode_label": list(self.episode_labels),
        })

        self.last_env_ids = [info.get("env_id", "") for info in infos]
        self.current_obs = list(next_obs)
        self.current_suffix = [str(info.get("suffix", "") or "") for info in infos]
        next_observations = {
            "text": self.build_text_obs(next_obs, init=False),
            "image": None,
            "anchor": next_obs.copy(),
        }
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
        return next_observations, rewards, dones, infos

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        Build prompt with optional recent history.
        """
        def _escape_braces(text: str) -> str:
            # Observations/actions often contain `\boxed{...}` which breaks `str.format`.
            return text.replace("{", "{{").replace("}", "}}")

        postprocess_text_obs: List[str] = []
        episode_cap = int(self.episode_max_steps or self.config.env.max_steps)

        if not init and self.config.env.history_length > 0:
            action_histories, valid_lens = self.memory.fetch(
                self.config.env.history_length,
                obs_key="text_obs",
                action_key="action",
                episode_key="episode_id",
                episode_step_key="episode_step",
                episode_label_key="episode_label",
            )
        else:
            action_histories = ["" for _ in range(len(text_obs))]
            valid_lens = [0 for _ in range(len(text_obs))]

        for i in range(len(text_obs)):
            env_id = self.last_env_ids[i] if i < len(self.last_env_ids) else ""
            suffix = self.current_suffix[i] if i < len(self.current_suffix) else ""
            safe_obs = _escape_braces(str(text_obs[i]))
            safe_suffix = _escape_braces(str(suffix))
            safe_history = _escape_braces(str(action_histories[i])) if i < len(action_histories) else ""

            if init:
                obs_i = GEM_TEMPLATE_MULTI_EPISODE_INIT.format(
                    episode_cap=episode_cap,
                    env_id=env_id,
                    current_observation=safe_obs,
                    task_suffix=safe_suffix,
                )
            else:
                obs_i = GEM_TEMPLATE_MULTI_EPISODE.format(
                    episode_cap=episode_cap,
                    env_id=env_id,
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=safe_history,
                    current_episode=self.episode_ids[i] + 1,
                    current_step=self.episode_step_ids[i] + 1,
                    current_observation=safe_obs,
                    task_suffix=safe_suffix,
                )
            postprocess_text_obs.append(obs_i)
        return postprocess_text_obs

    def soft_reset(self, env_indices, prev_infos=None):
        """
        Reset specified envs to their current task instances.
        IMPORTANT: this does NOT clear agent memory/history (multi-episode rollout).
        """
        if not env_indices:
            return {}, {}

        env_indices = [int(idx) for idx in env_indices]
        obs_map, info_map = self.envs.soft_reset(env_indices)

        # Episode bookkeeping (mirror ALFWorld multi-episode semantics)
        prev_infos = prev_infos or [None] * len(self.current_obs)
        for idx in env_indices:
            reason = (prev_infos[idx] or {}).get("multi_episode_soft_reset_reason", "success")
            last_episode_steps = self.episode_step_ids[idx]
            prev_episode = self.episode_ids[idx] + 1
            self.episode_ids[idx] += 1
            self.episode_step_ids[idx] = 0
            if self.multi_episode_enabled:
                episode_cap = int(self.episode_max_steps or self.config.env.max_steps)
                if reason == "success":
                    label = f"previous episode {prev_episode} succeeded in {last_episode_steps} step(s)"
                elif reason == "internal_max_turns":
                    label = f"previous episode {prev_episode} reached internal max turns in {last_episode_steps} step(s) without success"
                elif reason == "terminal":
                    label = f"previous episode {prev_episode} ended (failure/format error) in {last_episode_steps} step(s) without success"
                else:
                    label = f"previous episode {prev_episode} reached {last_episode_steps}/{episode_cap} step(s) without success"
                if self.memory and len(self.memory[idx]) > 0:
                    self.memory[idx][-1]["episode_label"] = label
                self.episode_labels[idx] = ""

        # Update cached current observations and env ids.
        for idx in env_indices:
            self.current_obs[idx] = obs_map[idx]
            self.last_env_ids[idx] = info_map[idx].get("env_id", self.last_env_ids[idx])
            self.current_suffix[idx] = str(info_map[idx].get("suffix", "") or "")

        # Build full prompts (including history) and return only updates.
        full_text_obs = self.build_text_obs(self.current_obs, init=False)
        obs_updates = {"text": {}, "image": {}, "anchor": {}}
        for idx in env_indices:
            obs_updates["text"][idx] = full_text_obs[idx]
            obs_updates["anchor"][idx] = self.current_obs[idx]

        return obs_updates, info_map

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        """
        Add GEM per-env_id metrics in addition to overall success_rate.
        """
        # Find the last entry with active masks
        last_active_i = None
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            if total_batch_list[batch_idx][i].get("active_masks"):
                last_active_i = i
                break
        if last_active_i is None:
            success["success_rate"].append(0.0)
            return

        final_info = total_infos[batch_idx][last_active_i]
        won_value = float(final_info.get("won", 0.0))
        success["success_rate"].append(won_value)

        env_id = str(final_info.get("env_id", "unknown"))
        env_key = env_id.replace(":", "_").replace("-", "_")
        success.setdefault(f"{env_key}_success_rate", []).append(won_value)


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        prompt_type = getattr(config.env, 'prompt_type', 'vanilla')
        if prompt_type == 'chat':
            from agent_system.memory import AlfWorldChatMemory
            self.memory = AlfWorldChatMemory(stripped_template=WEBSHOP_CHAT_USER_OBS_STRIPPED)
        else:
            self.memory = SimpleMemory()
        # Multi-episode config
        multi_episode_cfg = getattr(config.env, "multi_episode_rollout", None)
        self.multi_episode_enabled = bool(getattr(multi_episode_cfg, "enable", False)) if multi_episode_cfg else False
        self.episode_max_steps = getattr(multi_episode_cfg, "episode_max_steps", None) if multi_episode_cfg else None
        self.enable_reflection = bool(getattr(multi_episode_cfg, "enable_reflection", False)) if multi_episode_cfg else False
        # Support summary/vanilla prompt variants like ALFWorld
        self.prompt_init, self.prompt_history, self.keep_known_and_unknown = select_prompt_variant(
            config,
            WEBSHOP_TEMPLATE_NO_HIS,
            WEBSHOP_TEMPLATE,
            WEBSHOP_TEMPLATE_NO_HIS_SUMMARY,
            WEBSHOP_TEMPLATE_SUMMARY,
        )
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        self.pre_text_obs = obs
        self.current_infos = infos
        self.memory.reset(batch_size=len(infos))
        # Multi-episode tracking
        self.episode_ids = [0] * len(obs)
        self.episode_step_ids = [0] * len(obs)
        self.episode_labels = [""] * len(obs)
        self.prev_episode_labels = [""] * len(obs)
        self._reflection_pending = [False] * len(obs)

        if getattr(self.config.env, 'prompt_type', 'vanilla') == 'chat':
            full_text_obs = self._build_chat_obs(obs, infos, init=True)
        else:
            full_text_obs = self.build_text_obs(obs, infos, init=True)

        observations = {'text': full_text_obs, 'image': None, 'anchor': obs.copy()}
        return observations, infos

    def step(self, text_actions: List[str]):
        # extract known and unknown before projection if using summary prompts
        if self.keep_known_and_unknown:
            known_information, unknown_information = extract_known_and_unknown(text_actions)
        # Save raw model responses before projection
        prompt_type = getattr(self.config.env, 'prompt_type', 'vanilla')
        if prompt_type == 'chat':
            raw_responses = list(text_actions)

        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        next_obs = self.format_obs(next_obs)
        self.current_infos = infos

        for idx in range(len(actions)):
            if not self._reflection_pending[idx]:
                self.episode_step_ids[idx] += 1

        if prompt_type == 'chat':
            # Skip chat record storage for reflection-pending envs
            if any(self._reflection_pending):
                self._store_chat_record_partial(next_obs, infos, raw_responses)
            else:
                self._store_chat_record(next_obs, infos, raw_responses)
            text_field = self._build_chat_obs(next_obs, infos)
        else:
            self.memory.store({
                'text_obs': self.pre_text_obs,
                'action': actions,
                'episode_id': list(self.episode_ids),
                'episode_step': list(self.episode_step_ids),
                'episode_label': list(self.episode_labels),
            })
            if self.keep_known_and_unknown:
                text_field = self.build_text_obs_with_known_and_unknown(
                    next_obs, infos, known_information, unknown_information
                )
            else:
                text_field = self.build_text_obs(next_obs, infos)

        self.pre_text_obs = next_obs

        next_observations = {
            'text': text_field,
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def soft_reset(self, env_indices: List[int], prev_infos: List[Dict[str, Any]]):
        if not env_indices:
            return {}, {}

        env_indices = [int(idx) for idx in env_indices]
        obs_map, info_map = self.envs.soft_reset(env_indices)

        for idx in env_indices:
            # Format raw obs (strip task prefix)
            raw_text = obs_map[idx]
            parts = raw_text.split(" [SEP] ")
            try:
                task_index = parts.index(self.tasks[idx])
                formatted = " [SEP] ".join(f"'{p}'" for p in parts[task_index + 1:])
            except Exception:
                formatted = raw_text
            obs_map[idx] = formatted
            self.pre_text_obs[idx] = formatted
            self.current_infos[idx] = info_map[idx]

            # Episode bookkeeping
            last_episode_steps = self.episode_step_ids[idx]
            if last_episode_steps == 0 and self.memory and len(self.memory[idx]) > 0:
                last_episode_steps = int(self.memory[idx][-1].get("episode_step", 0))
            reason = prev_infos[idx].get("multi_episode_soft_reset_reason", "success")
            prev_episode = self.episode_ids[idx] + 1
            self.episode_ids[idx] += 1
            self.episode_step_ids[idx] = 0

            if self.multi_episode_enabled:
                episode_cap = int(self.episode_max_steps or self.config.env.max_steps)
                if reason == "success":
                    label = f"previous episode {prev_episode} succeeded in {last_episode_steps} step(s)"
                else:
                    label = (
                        f"previous episode {prev_episode} reached {last_episode_steps}/{episode_cap} step(s) "
                        f"without success"
                    )
                if self.memory and len(self.memory[idx]) > 0:
                    self.memory[idx][-1]["episode_label"] = label
                self.prev_episode_labels[idx] = label
                self.episode_labels[idx] = ""
            else:
                self.episode_labels[idx] = ""
                self.prev_episode_labels[idx] = ""

        prompt_type = getattr(self.config.env, 'prompt_type', 'vanilla')
        if prompt_type == 'chat':
            for idx in env_indices:
                available_actions = self.format_avail_actions(info_map[idx]['available_actions'])
                reformatted = "\n".join(f"'{s}'," for s in available_actions)
                user_text = WEBSHOP_CHAT_USER_OBS.format(
                    task_description=self.tasks[idx],
                    current_observation=self.pre_text_obs[idx],
                    available_actions=reformatted,
                )
                self.memory.store_single(idx, {
                    'user_text': user_text,
                    'assistant_text': None,
                    'raw_obs': self.pre_text_obs[idx],
                    'admissible_actions': reformatted,
                    'episode_id': self.episode_ids[idx],
                    'episode_step': self.episode_step_ids[idx],
                    'episode_label': self.episode_labels[idx],
                })
            system_prompts = self._build_chat_system_prompts(len(self.pre_text_obs))
            full_text_obs = self.memory.build_message_history(system_prompts)
        else:
            full_text_obs = self.build_text_obs(self.pre_text_obs, self.current_infos)

        obs_updates = {"text": {}, "image": {}, "anchor": {}}
        for idx in env_indices:
            obs_updates["text"][idx] = full_text_obs[idx]
            obs_updates["anchor"][idx] = self.pre_text_obs[idx]
            obs_updates["image"][idx] = None

        return obs_updates, info_map

    def build_reflection_obs(self, env_indices: List[int], next_obs: Dict, prev_infos: List[Dict]):
        """Append reflection prompt to terminal observation for completed episodes."""
        for idx in env_indices:
            self._reflection_pending[idx] = True
            terminal_obs = self.pre_text_obs[idx]
            episode_result = "succeeded" if prev_infos[idx].get("won", False) else "did not succeed"

            user_text = (
                f"Your current observation is: {terminal_obs}\n"
                f"Episode result: {episode_result}.\n\n"
                f"{WEBSHOP_CHAT_REFLECTION_PROMPT}"
            )

            self.memory.store_single(idx, {
                'user_text': user_text,
                'assistant_text': None,
                'raw_obs': terminal_obs,
                'admissible_actions': '',
                'episode_id': self.episode_ids[idx],
                'episode_step': self.episode_step_ids[idx],
                'episode_label': '',
                'is_reflection': True,
            })

        system_prompts = self._build_chat_system_prompts(len(self.pre_text_obs))
        full_text_obs = self.memory.build_message_history(system_prompts)

        for idx in env_indices:
            next_obs["text"][idx] = full_text_obs[idx]

    def consume_reflection(self, env_indices: List[int], text_actions: List[str], prev_infos: List[Dict]):
        """Store reflection responses in memory and soft_reset to start new episodes."""
        for idx in env_indices:
            last_rec = self.memory[idx][-1]
            assert last_rec.get('is_reflection', False), \
                f"Expected reflection record for env {idx}, got {last_rec.keys()}"
            last_rec['assistant_text'] = text_actions[idx]
            self._reflection_pending[idx] = False

        infos_for_reset = [None] * len(self.pre_text_obs)
        for idx, info in zip(env_indices, prev_infos):
            infos_for_reset[idx] = info

        return self.soft_reset(env_indices, infos_for_reset)

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks

    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs

    def format_obs_single(self, raw_text: str, task: str) -> str:
        """Format a single raw observation (used by soft_reset)."""
        parts = raw_text.split(" [SEP] ")
        try:
            index = parts.index(task)
            return " [SEP] ".join(f"'{p}'" for p in parts[index + 1:])
        except Exception:
            return raw_text

    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions

    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        memory_contexts = valid_lens = None
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action",
                    episode_key="episode_id" if self.multi_episode_enabled else None,
                    episode_step_key="episode_step" if self.multi_episode_enabled else None)

        episode_cap = int(self.episode_max_steps or self.config.env.max_steps) if self.multi_episode_enabled else None

        for i in range(len(text_obs)):

            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if self.multi_episode_enabled:
                previous_label = self.prev_episode_labels[i]
                if not previous_label and self.memory and len(self.memory[i]) > 0:
                    previous_label = self.memory[i][-1].get("episode_label", "")
                if init or self.config.env.history_length <= 0:
                    obs = WEBSHOP_TEMPLATE_MULTI_EPISODE_INIT.format(
                        episode_cap=episode_cap,
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions,
                    )
                else:
                    history_text = memory_contexts[i] if memory_contexts is not None else ""
                    current_ep = self.episode_ids[i] + 1
                    if self.episode_step_ids[i] == 0:
                        if not history_text:
                            if previous_label:
                                history_text = (
                                    f"--- Previous episode result: {previous_label} ---\n"
                                    f"--- Episode {current_ep} start ---"
                                )
                            else:
                                history_text = f"--- Episode {current_ep} start ---"
                        elif f"Episode {current_ep} start" not in history_text:
                            history_text = f"{history_text}\n--- Episode {current_ep} start ---"
                    obs = WEBSHOP_TEMPLATE_MULTI_EPISODE.format(
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=history_text,
                        current_episode=current_ep,
                        current_step=self.episode_step_ids[i] + 1,
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions,
                        episode_cap=episode_cap,
                    )
            else:
                if init or self.config.env.history_length <= 0:
                    obs = self.prompt_init.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )
                else:
                    obs = self.prompt_history.format(
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )
                    if len(obs) > 13000:
                        print(f"Warning len(obs)={len(obs)} is too long")
                        obs = self.prompt_init.format(
                            task_description=self.tasks[i],
                            current_observation=text_obs[i],
                            available_actions=reformatted_available_actions
                        )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def build_text_obs_with_known_and_unknown(self, text_obs: List[str], infos: List[List[str]], known_information: List[str], unknown_information: List[str]) -> List[str]:
        postprocess_text_obs = []
        if self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")

        for i in range(len(text_obs)):
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if self.config.env.history_length <= 0:
                obs = self.prompt_init.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = self.prompt_history.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    known_information=known_information[i],
                    unknown_information=unknown_information[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )

            if len(obs) > 13000:
                print(f"Warning len(obs)={len(obs)} is too long")
                obs = self.prompt_init.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    # ---- Chat (ORBIT-style multi-turn) helpers ---- #

    def _build_chat_system_prompts(self, batch_size: int) -> List[str]:
        if self.multi_episode_enabled:
            episode_cap = int(self.episode_max_steps or self.config.env.max_steps)
            return [WEBSHOP_CHAT_SYSTEM_PROMPT.format(episode_cap=episode_cap)] * batch_size
        return [WEBSHOP_CHAT_SYSTEM_PROMPT_SINGLE] * batch_size

    def _build_chat_obs(
        self,
        text_obs: List[str],
        infos: List[Dict[str, Any]],
        init: bool = False,
    ) -> List[List[Dict]]:
        if init:
            user_texts, raw_obs_list, admissible_strs = [], [], []
            for i in range(len(text_obs)):
                available_actions = self.format_avail_actions(infos[i]['available_actions'])
                reformatted = "\n".join(f"'{s}'," for s in available_actions)
                user_text = WEBSHOP_CHAT_USER_OBS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted,
                )
                user_texts.append(user_text)
                raw_obs_list.append(text_obs[i])
                admissible_strs.append(reformatted)

            self.memory.store({
                'user_text': user_texts,
                'assistant_text': [None] * len(text_obs),
                'raw_obs': raw_obs_list,
                'admissible_actions': admissible_strs,
                'episode_id': list(self.episode_ids),
                'episode_step': list(self.episode_step_ids),
                'episode_label': list(self.episode_labels),
            })

        system_prompts = self._build_chat_system_prompts(len(text_obs))
        return self.memory.build_message_history(system_prompts)

    def _store_chat_record(self, text_obs: List[str], infos: List[Dict[str, Any]], raw_responses: List[str]):
        user_texts, admissible_strs = [], []
        for i in range(len(text_obs)):
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted = "\n".join(f"'{s}'," for s in available_actions)
            user_text = WEBSHOP_CHAT_USER_OBS.format(
                task_description=self.tasks[i],
                current_observation=text_obs[i],
                available_actions=reformatted,
            )
            user_texts.append(user_text)
            admissible_strs.append(reformatted)

        self.memory.store({
            'user_text': user_texts,
            'assistant_text': raw_responses,
            'raw_obs': list(text_obs),
            'admissible_actions': admissible_strs,
            'episode_id': list(self.episode_ids),
            'episode_step': list(self.episode_step_ids),
            'episode_label': list(self.episode_labels),
        })

    def _store_chat_record_partial(self, text_obs: List[str], infos: List[Dict[str, Any]], raw_responses: List[str]):
        """Store chat records only for envs NOT in reflection-pending state."""
        for i in range(len(text_obs)):
            if self._reflection_pending[i]:
                continue
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted = "\n".join(f"'{s}'," for s in available_actions)
            user_text = WEBSHOP_CHAT_USER_OBS.format(
                task_description=self.tasks[i],
                current_observation=text_obs[i],
                available_actions=reformatted,
            )
            self.memory.store_single(i, {
                'user_text': user_text,
                'assistant_text': raw_responses[i],
                'raw_obs': text_obs[i],
                'admissible_actions': reformatted,
                'episode_id': self.episode_ids[i],
                'episode_step': self.episode_step_ids[i],
                'episode_label': self.episode_labels[i],
            })

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score_not_success_rate'].append(score_value)
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs

class WebVoyagerEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = WebVoyagerMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        obs_list, infos = self.envs.reset()

        # Initialize memory buffer
        batch_size = len(obs_list)
        self.memory.reset(batch_size=batch_size)

        # Build initial user messages and store
        initial_user_texts: List[str] = []
        images_batch: List[Any] = []
        image_paths: List[Any] = []
        urls: List[Any] = []
        trees: List[Any] = []
        warns: List[Any] = []
        pdfs: List[Any] = []
        fails: List[Any] = []
        
        for obs in obs_list:
            task_goal = obs.get('task_ques', '')
            url = obs.get('starting_url', '')
            tree = obs.get('ac_tree', '')
            img_path = obs.get('image')

            init_text = WEBVOYAGER_PROMPT_TEMPLATE["initial"].format(
                task_goal=task_goal,
                url=url,
                accessibility_tree=tree or ""
            )
            # # Append a single image placeholder if we have a screenshot
            # if img_path:
            #     init_text = init_text + "\n\n<image>"

            initial_user_texts.append(init_text)
            images_batch.append([img_path] if img_path else None)
            image_paths.append(img_path)
            urls.append(url)
            trees.append(tree)
            warns.append(obs.get('warn_obs', ''))
            pdfs.append(obs.get('pdf_obs', ''))
            fails.append(obs.get('fail_obs', ''))

        self.memory.store({
            "user_text": initial_user_texts,
            "assistant_text": [None] * batch_size,
            "image_path": image_paths,
            "url": urls,
            "tree": trees,
            "warn_obs": warns,
            "pdf_obs": pdfs,
            "fail_obs": fails,
            "action": [""] * batch_size,
        })

        # Emit chat-formatted messages for the LLM; rollout flattens and extracts images
        messages_per_env = self.memory.build_message_history(history_length=3, max_images=2)
        observations = {
            'text': messages_per_env,
            'image': None,
            'anchor': obs_list
        }
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        # Build next user messages (observation texts) and store together with assistant_text
        next_user_texts: List[str] = []
        images_batch: List[Any] = []
        image_paths: List[Any] = []
        urls: List[Any] = []
        trees: List[Any] = []
        warns: List[Any] = []
        pdfs: List[Any] = []
        fails: List[Any] = []

        for obs in next_obs:
            task_goal = obs.get('task_ques', '')
            url = obs.get('url', '') # current url
            tree = obs.get('ac_tree', '')
            img_path = obs.get('image')
            pdf_obs = obs.get('pdf_obs', '')

            if pdf_obs:
                user_text = WEBVOYAGER_PROMPT_TEMPLATE["pdf_observation"].format(
                    task_goal=task_goal,
                    pdf_obs=pdf_obs
                )
            else:
                user_text = WEBVOYAGER_PROMPT_TEMPLATE["observation"].format(
                    task_goal=task_goal,
                    url=url,
                    accessibility_tree=tree or ""
                )

            # if img_path:
            #     user_text = user_text + "\n\n<image>"

            next_user_texts.append(user_text)
            images_batch.append([img_path] if img_path else None)
            image_paths.append(img_path)
            urls.append(url)
            trees.append(tree)
            warns.append(obs.get('warn_obs', ''))
            pdfs.append(pdf_obs)
            fails.append(obs.get('fail_obs', ''))

        # Store assistant_text (raw model outputs) and the newly built user_texts
        self.memory.store({
            "user_text": next_user_texts,
            "assistant_text": text_actions,
            "image_path": image_paths,
            "url": urls,
            "tree": trees,
            "warn_obs": warns,
            "pdf_obs": pdfs,
            "fail_obs": fails,
            "action": text_actions,
        })

        # Emit chat-formatted messages for the LLM; rollout flattens and extracts images
        messages_per_env = self.memory.build_message_history(history_length=3, max_images=2)
        next_observations = {
            'text': messages_per_env,
            'image': None,
            'anchor': next_obs
        }

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos




def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': 'eval_out_of_distribution', # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webvoyager" in config.env.env_name.lower():
        from agent_system.environments.env_package.webvoyager import build_webvoyager_envs, webvoyager_projection
        env_kwargs = {}
        config_path = os.path.join(os.path.dirname(__file__), 'env_package/webvoyager/configs/webvoyager_configs.yaml')
        _envs = build_webvoyager_envs(config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_webvoyager_envs(config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(webvoyager_projection)
        envs = WebVoyagerEnvironmentManager(_envs, projection_f, config)
        val_envs = WebVoyagerEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "gem" in config.env.env_name.lower():
        from agent_system.environments.env_package.gem.builder import (
            build_gem_envs,
            gem_projection,
            GEM_TASK_POOL_TRAIN,
            GEM_TASK_POOL_EVAL,
        )
        gem_cfg = getattr(config.env, "gem", None)
        # Default: use the fixed task pools defined in gem.builder.
        use_default_pool = getattr(gem_cfg, "use_default_pool", True) if gem_cfg else True
        if gem_cfg and getattr(gem_cfg, "env_ids", None) is not None:
            env_ids = list(getattr(gem_cfg, "env_ids"))
        elif use_default_pool:
            env_ids = sorted({t["env_id"] for t in (GEM_TASK_POOL_TRAIN + GEM_TASK_POOL_EVAL)})
        else:
            env_ids = ["game:GuessTheNumber-v0-easy"]
        task_pool_train = getattr(gem_cfg, "task_pool_train", None) if gem_cfg else None
        task_pool_val = getattr(gem_cfg, "task_pool_val", None) if gem_cfg else None
        max_steps = getattr(config.env, "max_steps", None)

        _envs = build_gem_envs(
            env_ids=env_ids,
            seed=config.env.seed,
            env_num=config.data.train_batch_size,
            group_n=group_n,
            max_steps=max_steps,
            resources_per_worker=resources_per_worker,
            task_pool=task_pool_train,
            use_default_pool=use_default_pool and task_pool_train is None,
            is_train=True,
        )
        _val_envs = build_gem_envs(
            env_ids=env_ids,
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            group_n=1,
            max_steps=max_steps,
            resources_per_worker=resources_per_worker,
            task_pool=task_pool_val,
            use_default_pool=use_default_pool and task_pool_val is None,
            is_train=False,
        )

        projection_f = partial(gem_projection)
        envs = GemEnvironmentManager(_envs, projection_f, config)
        val_envs = GemEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webarena" in config.env.env_name.lower():
        # use webvoyager envs to build webarena envs for now
        from agent_system.environments.env_package.webvoyager import build_webvoyager_envs, webvoyager_projection
        env_kwargs = {}
        config_path = os.path.join(os.path.dirname(__file__), 'env_package/webvoyager/configs/webarena_configs.yaml')
        _envs = build_webvoyager_envs(config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_webvoyager_envs(config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(webvoyager_projection)
        envs = WebVoyagerEnvironmentManager(_envs, projection_f, config)
        val_envs = WebVoyagerEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)