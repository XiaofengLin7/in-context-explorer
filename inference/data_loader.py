"""
Simple JSONL data loader for prompts/messages used with vLLMEngine.

Supported JSONL line formats:
- {"prompt": "<string>"}                           -> wrapped as [{"role":"user","content": "<string>"}]
- {"messages": [{"role":"system","content":"..."}, {"role":"user","content":"..."}]}
"""
from __future__ import annotations

import json
from typing import Any, Dict, Generator, List, Optional, Union

Messages = List[Dict[str, str]]


def parse_role_tagged_prompt(prompt: str) -> Messages:
    """
    Parse a single string that contains role markers on their own lines:
        system
        <system text...>
        user
        <user text...>
        assistant
        <assistant text...>
    Returns a list of messages preserving the order encountered.
    Unknown lines before any role tag are treated as user content.
    """
    lines = prompt.splitlines()
    roles = {"system", "user", "assistant"}
    current_role: Optional[str] = None
    buffers: List[Dict[str, Any]] = []

    def _push(role: str, content_lines: List[str]):
        content = "\n".join(content_lines).strip()
        if content:
            buffers.append({"role": role, "content": content})

    acc: List[str] = []
    for line in lines:
        tag = line.strip()
        if tag in roles:
            # close previous
            if current_role is not None:
                _push(current_role, acc)
                acc = []
            current_role = tag
        else:
            acc.append(line)
    # finalize
    if current_role is not None:
        _push(current_role, acc)
    else:
        # no tagged roles found -> treat entire content as a user message
        if prompt.strip():
            buffers.append({"role": "user", "content": prompt.strip()})

    # if the last assistant message is empty or trailing marker only, it won't be included
    # keep all roles that have content
    return [{"role": m["role"], "content": m["content"]} for m in buffers]


def wrap_prompt_to_messages(prompt: str) -> Messages:
    """
    Convert a plain prompt string into messages. If it contains role tags ('system', 'user', 'assistant')
    on their own lines, split accordingly; otherwise, return a single user message.
    """
    tagged = parse_role_tagged_prompt(prompt)
    if tagged:
        return tagged
    return [{"role": "user", "content": prompt}]


def iter_jsonl(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """
    Yields one JSON object per line from a .jsonl file.
    Ignores blank lines and lines that fail JSON parsing.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def iter_steps_from_jsonl(
    file_path: str,
    *,
    traj_key: str = "traj_uid",
    steps_key: str = "steps",
    prompt_key: str = "prompt",
) -> Generator[Dict[str, Any], None, None]:
    """
    Yields per-step records as:
        {
          "traj_uid": <str>,
          "step": <int>,
          "messages": <Messages>
        }
    The messages are derived from each step's prompt (role-tagged string is supported).
    """
    for obj in iter_jsonl(file_path):
        traj_uid = obj.get(traj_key)
        steps = obj.get(steps_key, [])
        if not isinstance(steps, list):
            continue
        for st in steps:
            if not isinstance(st, dict):
                continue
            step_id = st.get("step")
            prompt = st.get(prompt_key)
            if prompt is None or not isinstance(prompt, str):
                continue
            msgs = wrap_prompt_to_messages(prompt)
            yield {"traj_uid": traj_uid, "step": step_id, "messages": msgs}


