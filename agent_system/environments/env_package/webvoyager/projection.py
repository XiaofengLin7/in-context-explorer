from typing import List, Tuple, Dict, Any
import re
import json


ALLOWED_ACTIONS = {"click", "type", "scroll", "wait", "goback", "google", "answer"}


def _extract_action_block(text: str) -> str:
    """Return the content on the line starting with "Action:" (case-insensitive).

    The prompt specifies concise commands on the Action line. We do not support
    JSON payloads or <action>...</action> blocks.

    Returns an empty string if no action payload is found.
    """
    m = re.search(r"(?mi)^Action\s*:\s*(.+)$", text)
    return m.group(1).strip() if m else ""


def _has_think_block(text: str) -> bool:
    """Return True if the output contains a reasoning block.

    Accepts either an explicit <think>...</think> block or a line beginning with
    "Thought:" to match the prompt's required format.
    """
    return bool(
        re.search(r"<think>[\s\S]*?</think>", text, flags=re.IGNORECASE)
        or re.search(r"(?mi)^Thought\s*:\s*", text)
    )


def _contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _parse_json_like(s: str) -> Dict[str, Any] | None:
    """Deprecated: JSON parsing is no longer supported."""
    return None


def _fallback_parse(s: str) -> Dict[str, Any] | None:
    """Fallback parser for concise command formats, e.g.:
    - Click [12]
    - Type [3]; [hello]
    - Scroll [WINDOW]; [down]
    - Wait
    - GoBack
    - Google
    - ANSWER; [final answer]
    Returns normalized action dict or None on failure.
    """
    lower = s.strip()
    # Click [n]
    m = re.search(r"^click\s*\[\s*(\d+)\s*\]$", lower, re.IGNORECASE)
    if m:
        return {"action_key": "click", "element": int(m.group(1))}

    # Type [n]; [text]  (legacy, with semicolon)
    m = re.search(r"^type\s*\[\s*(\d+)\s*\]\s*;\s*\[(.*?)\]$", lower, re.IGNORECASE | re.DOTALL)
    if m:
        return {"action_key": "type", "element": int(m.group(1)), "content": m.group(2).strip()}

    # Type [n] [text]  (prompt-specified, without semicolon)
    m = re.search(r"^type\s*\[\s*(\d+)\s*\]\s*\[\s*(.*?)\s*\]$", lower, re.IGNORECASE | re.DOTALL)
    if m:
        return {"action_key": "type", "element": int(m.group(1)), "content": m.group(2).strip()}

    # Scroll [WINDOW|n]; [up|down]  (legacy)
    m = re.search(r"^scroll\s*\[\s*(window|\d+)\s*\]\s*;\s*\[(up|down)\]$", lower, re.IGNORECASE)
    if m:
        elem = -1 if m.group(1).lower() == "window" else int(m.group(1))
        return {"action_key": "scroll", "element": elem, "content": m.group(2).lower()}

    # Scroll [up|down]  (prompt-specified, window by default)
    m = re.search(r"^scroll\s*\[\s*(up|down)\s*\]$", lower, re.IGNORECASE)
    if m:
        return {"action_key": "scroll", "element": -1, "content": m.group(1).lower()}

    # Wait
    if re.fullmatch(r"wait", lower, re.IGNORECASE):
        return {"action_key": "wait"}

    # GoBack
    if re.fullmatch(r"goback", lower, re.IGNORECASE):
        return {"action_key": "goback"}

    # Google
    if re.fullmatch(r"google", lower, re.IGNORECASE):
        return {"action_key": "google"}

    # ANSWER; [text]  (legacy)
    m = re.search(r"^answer\s*;\s*\[(.*?)\]$", lower, re.IGNORECASE | re.DOTALL)
    if m:
        return {"action_key": "answer", "answer_content": m.group(1).strip()}

    # ANSWER [text]  (prompt-specified)
    m = re.search(r"^answer\s*\[\s*(.*?)\s*\]$", lower, re.IGNORECASE | re.DOTALL)
    if m:
        return {"action_key": "answer", "answer_content": m.group(1).strip()}

    return None


def _normalize_action(obj: Dict[str, Any]) -> Dict[str, Any] | None:
    """Deprecated: JSON normalization is no longer supported."""
    return None


def webvoyager_projection(actions: List[str]) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Project LLM outputs into structured WebVoyager actions.

    Input:
        actions: list of model text outputs, each should contain a single line
                starting with "Action:" followed by one concise command, and
                ideally a line starting with "Thought:" containing reasoning.
                Supported concise commands:
                - "Click [n]"
                - "Type [n] [text]" (legacy: "Type [n]; [text]")
                - "Scroll [up]" or "Scroll [down]" (legacy: "Scroll [WINDOW]; [up/down]")
                - "Wait"
                - "GoBack"
                - "Google"
                - "ANSWER [text]" (legacy: "ANSWER; [text]")

    Output:
        - normalized_actions: list of dicts matching WebVoyagerEnv.step expectations
        - valids: list[int] flags (1 valid, 0 invalid)
    """

    normalized_actions: List[Dict[str, Any]] = []
    valids: List[int] = [1] * len(actions)

    for i, raw in enumerate(actions):
        original = raw
        # Extract Action-line payload
        payload = _extract_action_block(raw)
        if not payload:
            valids[i] = 0
            normalized_actions.append({"action_key": "wait"})
            continue

        # Parse concise command formats only
        norm = _fallback_parse(payload)

        if norm is None:
            valids[i] = 0
            norm = {"action_key": "wait"}

        normalized_actions.append(norm)

        # Additional validity heuristics
        if not _has_think_block(original):
            valids[i] = 0
        if _contains_chinese(original):
            valids[i] = 0

    return normalized_actions, valids