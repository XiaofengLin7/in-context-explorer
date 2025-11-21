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

from typing import List, Dict, Any, Tuple, Optional
import re
from .base import BaseMemory

class SimpleMemory(BaseMemory):
    """
    Memory manager: responsible for storing & fetching per‑environment history records.
    """
    def __init__(self):
        self._data = None
        self.keys = None
        self.batch_size = 0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        """
        Store a new record (one step of history) for each environment instance.

        Args:
            record (Dict[str, List[Any]]):
                A dictionary where each key corresponds to a type of data 
                (e.g., 'text_obs', 'action'), and each value is a list of 
                length `batch_size`, containing the data for each environment.
        """
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def fetch(
        self,
        history_length: int,
        obs_key: str = "text_obs",
        action_key: str = "action",
    ) -> Tuple[List[str], List[int]]:
        """
        Fetch and format recent interaction history for each environment instance.
        Args:
            history_length (int):
                Maximum number of past steps to retrieve per environment.
            obs_key (str, default="text_obs"):
                The key name used to access the observation in stored records.
                For example: "text_obs" or "Observation", depending on the environment.
            action_key (str, default="action"):
                The key name used to access the action in stored records.
                For example: "action" or "Action".
        Returns:
            memory_contexts : List[str]
                A list of formatted action history strings for each environment.
            valid_lengths : List[int]
                A list of the actual number of valid history steps per environment.
        """
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec[action_key]
                obs = rec[obs_key]
                lines.append(
                    f"[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"
                )

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths

class WebVoyagerMemory(BaseMemory):
    """
    Memory manager for webvoyager tasks: responsible for storing & fetching
    Logging interaction history between agent and environment to provide context for the next action.
    """
    def __init__(self):
        self._data = None
        self.keys = None
        self.batch_size = 0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        """
        Store a new record (one step of history) for each environment instance.

        The expected record keys should be consistent across calls and include
        at least the observation and action fields used downstream. For high-fidelity
        reconstruction, records may include the following optional keys:
        - 'user_text': str — full observation text shown to the LLM
        - 'assistant_text': Optional[str] — assistant's previous response text
        - 'image_path': Optional[str] — screenshot path associated with the observation
        - 'url': Optional[str]
        - 'tree': Optional[str] — full accessibility tree (kept intact)
        - 'warn_obs', 'pdf_obs', 'fail_obs': Optional[str]
        - 'action': Optional[str] — parsed action line

        Args:
            record (Dict[str, List[Any]]):
                A dictionary where each key corresponds to a type of data and each
                value is a list of length `batch_size`, containing data for each
                environment at the current step.
        """
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def fetch(
        self,
        history_length: int,
        obs_key: str = "user_text",
        action_key: str = "action",
    ) -> Tuple[List[str], List[int]]:
        """
        Fetch and format recent interaction history for each environment instance.

        This returns compact strings primarily for logging or simpler models. For
        high-fidelity message reconstruction compatible with the LLM schema, use
        `build_message_history`.

        Args:
            history_length (int):
                Maximum number of past steps to retrieve per environment.
            obs_key (str, default="user_text"):
                Key used to access the observation text in stored records.
            action_key (str, default="action"):
                Key used to access the action in stored records.

        Returns:
            Tuple[List[str], List[int]]: (memory_contexts, valid_lengths)
                - memory_contexts: a list of formatted history strings per env
                - valid_lengths: number of valid history steps per env
        """
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec.get(action_key, "")
                obs = rec.get(obs_key, "")
                lines.append(
                    f"[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"
                )

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths

    def build_message_history(
        self,
        history_length: int = 3,
        max_images: int = 3,
    ) -> List[List[Dict[str, Any]]]:
        """
        Reconstruct a high-fidelity, per-environment message history compatible
        with the LLM message schema used by WebVoyager prompts.

        The returned structure per env is a chronological list of messages, where
        each message is a dict: {'role': <'user'|'assistant'>, 'content': [ ... ]}
        and 'content' is a list of blocks, where each block is either:
          - {'type': 'text', 'text': <str>}
          - {'type': 'image', 'source': {'type': 'path', 'path': <str>}}

        Behavior:
        - All stored records per env are included (no truncation by history_length).
        - Older user messages are compressed to: "Observation omitted for previous steps. See attachment for screenshot."
        - A global cap of `max_images` images is applied per env. The latest user message is always preserved
          with its full detailed observation (including accessibility_tree and image). Older messages are clipped
          to ensure the total image count (history + latest) does not exceed `max_images`, removing the oldest
          images first and cleaning the helper text accordingly.
        - If the immediately preceding assistant message contains an ANSWER action,
          the most recent user message is appended with the double-check guidance.

        Args:
            history_length (int): Number of recent records to include. Default is 3.
            max_images (int): Maximum number of images to keep across all messages. Default is 3.

        Returns:
            List[List[Dict[str, Any]]]: A list of message lists, one per environment.
        """
        all_env_histories: List[List[Dict[str, Any]]] = []

        for env_idx in range(self.batch_size):
            # Include all records to mirror environment behavior (do not drop prior user messages)
            history_records = self._data[env_idx]
            messages: List[Dict[str, Any]] = []

            # Build alternating assistant -> user messages for each record when available
            for rec in history_records:
                assistant_text: Optional[str] = rec.get("assistant_text")
                if assistant_text:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": assistant_text},
                            ],
                        }
                    )

                # Construct user message from stored observation text and optional image
                user_text: str = rec.get("user_text", "")
                user_msg_content: List[Dict[str, Any]] = [
                    {"type": "text", "text": user_text},
                ]
                image_path: Optional[str] = rec.get("image_path")
                if image_path:
                    user_msg_content.append(
                        {
                            "type": "image",
                            "source": {"type": "path", "path": image_path},
                        }
                    )

                messages.append({"role": "user", "content": user_msg_content})

            # Apply clipping to ensure exactly max_images images total, while preserving
            # the latest user message with its full detailed observation (including accessibility_tree and image).
            # Clip only the history before the latest message, accounting for the latest message's image.
            if len(messages) > 0:
                # Find the latest user message
                last_user_idx = None
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        last_user_idx = i
                        break
                
                if last_user_idx is not None and last_user_idx > 0:
                    # Count images in the latest user message
                    latest_msg = messages[last_user_idx]
                    latest_images = 0
                    if latest_msg.get("role") == "user":
                        content = latest_msg.get("content", [])
                        for item in content:
                            if item.get("type") and "image" in item.get("type"):
                                latest_images += 1
                    
                    # Clip history messages to (max_images - latest_images) images
                    # This ensures total images = latest_images + history_images <= max_images
                    history_msgs = messages[:last_user_idx]
                    latest_msgs = messages[last_user_idx:]
                    max_history_images = max(0, max_images - latest_images)
                    clipped_history = self._clip_messages_like_env(history_msgs, max_history_images)
                    messages = clipped_history + latest_msgs
                else:
                    # If no history or only one message, clip all messages normally
                    messages = self._clip_messages_like_env(messages, max_images)

            # ANSWER double-check guard: if previous assistant message (immediately
            # before the newest user message) contains ANSWER, append guidance
            if messages:
                # find last assistant before the last user
                last_user_idx = None
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        last_user_idx = i
                        break
                if last_user_idx is not None:
                    # search backwards for assistant
                    for j in range(last_user_idx - 1, -1, -1):
                        if messages[j].get("role") == "assistant":
                            blocks = messages[j].get("content", [])
                            assistant_text_block = blocks[0]["text"] if blocks and blocks[0].get("type") == "text" else ""
                            if (
                                "Action: ANSWER" in assistant_text_block
                                or "Action:\nANSWER" in assistant_text_block
                            ):
                                # append reminder to newest user message
                                if (
                                    messages[last_user_idx]["content"]
                                    and messages[last_user_idx]["content"][0].get("type") == "text"
                                ):
                                    messages[last_user_idx]["content"][0][
                                        "text"
                                    ] += (
                                        "\n\nImportant: You returned an answer in the last step. Let's pause, "
                                        "check the web page, and think again. If you still think the task is "
                                        "finished, double-check your answer, revise it if need, and return a final "
                                        "answer. If not, continue the task."
                                    )
                            break

            all_env_histories.append(messages)

        return all_env_histories

    def _clip_images(self, messages: List[Dict[str, Any]], max_images: int) -> List[Dict[str, Any]]:
        """
        Enforce a maximum number of images across all user messages. Oldest images
        are removed first. When an image is removed, the helper hint in the text
        is also cleaned by removing "See attachment for screenshot.".

        Args:
            messages (List[Dict[str, Any]]): Message list for a single environment.
            max_images (int): Maximum allowed images.

        Returns:
            List[Dict[str, Any]]: Updated messages with images clipped if needed.
        """
        # Gather all user images in chronological order
        image_positions: List[Tuple[int, int]] = []
        total_images = 0
        for i, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", [])
            for j, item in enumerate(content):
                if item.get("type") == "image":
                    image_positions.append((i, j))
                    total_images += 1

        if total_images <= max_images:
            return messages

        to_remove = total_images - max_images
        removed = 0
        for i, j in image_positions:
            if removed >= to_remove:
                break
            # Remove the image block
            if j < len(messages[i]["content"]):
                del messages[i]["content"][j]
                # Clean up helper text in the first text block if present
                if messages[i]["content"] and messages[i]["content"][0].get("type") == "text":
                    cleaned = (
                        messages[i]["content"][0]["text"].replace(
                            "See attachment for screenshot.", ""
                        ).strip()
                    )
                    messages[i]["content"][0]["text"] = cleaned
                removed += 1

        return messages

    def _clip_messages_like_env(self, messages: List[Dict[str, Any]], max_images: int) -> List[Dict[str, Any]]:
        """
        Rewrite user message texts to an omitted placeholder and enforce a max
        number of attached images, mirroring the environment's clip behavior.

        Rules:
        - For each user message's text block:
          - If it already contains "Observation omitted for previous steps.", leave it.
          - If it contains "Now solve the following task.", trim everything from
            "Screenshot of current viewpoint:" onward, then replace the whole text
            with the omitted placeholder.
          - Otherwise, replace with the omitted placeholder.
        - Count all image blocks across user messages. If over `max_images`,
          delete images in chronological order and remove the helper suffix
          "See attachment for screenshot." from the corresponding text block.
        """
        OMITTED = "Observation omitted for previous steps. See attachment for screenshot."

        # First pass: rewrite texts and count images
        img_count = 0
        for i, msg in enumerate(messages):
            if msg.get('role') != 'user':
                continue
            content = msg.get('content', [])
            for j, item in enumerate(content):
                if item.get('type') == 'text':
                    text = content[j].get('text', '')
                    if "Observation omitted for previous steps." in text:
                        continue
                    if "Now solve the following task." in text:
                        # If present, trim up to before the screenshot indicator
                        m = re.search(r"Screenshot of current viewpoint:", text)
                        if m:
                            messages[i]['content'][j]['text'] = text[: m.start()] + OMITTED
                        else:
                            messages[i]['content'][j]['text'] = OMITTED
                    else:
                        messages[i]['content'][j]['text'] = OMITTED
                elif item.get('type') and 'image' in item.get('type'):
                    img_count += 1

        # Second pass: clip images if exceeding max_images
        if img_count > max_images:
            to_remove = img_count - max_images
            for i, msg in enumerate(messages):
                if to_remove <= 0:
                    break
                if msg.get('role') != 'user':
                    continue
                content = msg.get('content', [])
                k = 0
                while k < len(content) and to_remove > 0:
                    if content[k].get('type') and 'image' in content[k].get('type'):
                        del content[k]
                        to_remove -= 1
                        # Clean helper suffix in the leading text block if present
                        if content and content[0].get('type') == 'text':
                            cleaned = content[0]['text'].replace(
                                'See attachment for screenshot.', ''
                            ).strip()
                            content[0]['text'] = cleaned
                        continue
                    k += 1

        return messages


class SearchMemory(BaseMemory):
    """
    Memory manager for search tasks: responsible for storing & fetching
    """
    def __init__(self):
        self._data = None
        self.keys = None
        self.batch_size = 0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        """
        Store a new record (one step of history) for each environment instance.

        Args:
            record (Dict[str, List[Any]]):
                A dictionary where each key corresponds to a type of data 
                (e.g., 'text_obs', 'action'), and each value is a list of 
                length `batch_size`, containing the data for each environment.
        """
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def fetch(
        self,
        history_length: int,
        obs_key: str,
        action_key: str,
    ) -> Tuple[List[str], List[int]]:
        """
        Fetch and format recent interaction history for each environment instance.
        Args:
            history_length (int):
                Maximum number of past steps to retrieve per environment.
            obs_key (str):
                The key name used to access the observation in stored records.
                For example: "text_obs" or "Observation", depending on the environment.
            action_key (str):
                The key name used to access the action in stored records.
                For example: "action" or "Action".
        Returns:
            memory_contexts : List[str]
                A list of formatted action history strings for each environment.
            valid_lengths : List[int]
                A list of the actual number of valid history steps per environment.
        """
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec[action_key]
                obs = rec[obs_key]
                lines.append(
                    f"Step {step_num}:{act} {obs}\n"
                )

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths