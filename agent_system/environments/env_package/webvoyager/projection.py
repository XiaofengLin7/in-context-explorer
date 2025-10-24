from typing import List, Tuple, Dict, Any
import re
import json


ALLOWED_ACTIONS = {"click", "type", "scroll", "wait", "goback", "google", "answer"}


def _extract_action_block(text: str) -> str:
    """Return the inner content of the first <action>...</action> block (case-insensitive).

    If not found, return an empty string.
    """
    m = re.search(r"<action>([\s\S]*?)</action>", text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _has_think_block(text: str) -> bool:
    return bool(re.search(r"<think>[\s\S]*?</think>", text, flags=re.IGNORECASE))


def _contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _parse_json_like(s: str) -> Dict[str, Any] | None:
    """Try to parse a JSON object from string s.

    - Strips code fences and trailing text.
    - Accepts both single/double quotes by normalizing to double when safe.
    Returns None on failure.
    """
    # Remove code fences if present
    s_clean = s.strip()
    if s_clean.startswith("```"):
        # Take the first fenced block content
        parts = s_clean.split("```", 2)
        if len(parts) >= 3:
            s_clean = parts[1] if "{" in parts[1] else parts[2]
        s_clean = s_clean.strip()

    # Trim anything after the first balanced JSON object if the model added prose
    # Heuristic: cut at last '}'
    if "}" in s_clean:
        s_clean = s_clean[: s_clean.rfind("}") + 1]

    try:
        return json.loads(s_clean)
    except Exception:
        # Simple normalization for single quotes if present and no embedded quotes
        try:
            if "'" in s_clean and '"' not in s_clean:
                return json.loads(s_clean.replace("'", '"'))
        except Exception:
            pass
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

    # Type [n]; [text]
    m = re.search(r"^type\s*\[\s*(\d+)\s*\]\s*;\s*\[(.*?)\]$", lower, re.IGNORECASE | re.DOTALL)
    if m:
        return {"action_key": "type", "element": int(m.group(1)), "content": m.group(2).strip()}

    # Scroll [WINDOW|n]; [up|down]
    m = re.search(r"^scroll\s*\[\s*(window|\d+)\s*\]\s*;\s*\[(up|down)\]$", lower, re.IGNORECASE)
    if m:
        elem = -1 if m.group(1).lower() == "window" else int(m.group(1))
        return {"action_key": "scroll", "element": elem, "content": m.group(2).lower()}

    # Wait
    if re.fullmatch(r"wait", lower, re.IGNORECASE):
        return {"action_key": "wait"}

    # GoBack
    if re.fullmatch(r"goback", lower, re.IGNORECASE):
        return {"action_key": "goback"}

    # Google
    if re.fullmatch(r"google", lower, re.IGNORECASE):
        return {"action_key": "google"}

    # ANSWER; [text]
    m = re.search(r"^answer\s*;\s*\[(.*?)\]$", lower, re.IGNORECASE | re.DOTALL)
    if m:
        return {"action_key": "answer", "answer_content": m.group(1).strip()}

    return None


def _normalize_action(obj: Dict[str, Any]) -> Dict[str, Any] | None:
    """Normalize a parsed object to the env's expected schema.

    Expected keys by action_key:
    - click: element:int
    - type: element:int, content:str
    - scroll: element:int (-1 for window), content in {up,down}
    - wait/goback/google: no extras
    - answer: answer_content:str
    Returns None if invalid or unsupported.
    """
    if not isinstance(obj, dict):
        return None

    # Key normalization
    obj_norm: Dict[str, Any] = {k.lower(): v for k, v in obj.items()}
    if "action" in obj_norm and "action_key" not in obj_norm:
        obj_norm["action_key"] = obj_norm.pop("action")
    if "idx" in obj_norm and "element" not in obj_norm:
        obj_norm["element"] = obj_norm.pop("idx")
    if "number" in obj_norm and "element" not in obj_norm:
        obj_norm["element"] = obj_norm.pop("number")
    if "direction" in obj_norm and "content" not in obj_norm:
        obj_norm["content"] = obj_norm.pop("direction")
    if "answer" in obj_norm and "answer_content" not in obj_norm:
        obj_norm["answer_content"] = obj_norm.pop("answer")

    action_key = str(obj_norm.get("action_key", "")).lower()
    if action_key not in ALLOWED_ACTIONS:
        return None

    # Validate per action
    if action_key == "click":
        try:
            obj_norm["element"] = int(obj_norm["element"])  # type: ignore[index]
        except Exception:
            return None
        return {"action_key": "click", "element": obj_norm["element"]}

    if action_key == "type":
        try:
            obj_norm["element"] = int(obj_norm["element"])  # type: ignore[index]
            content = str(obj_norm.get("content", ""))
        except Exception:
            return None
        return {"action_key": "type", "element": obj_norm["element"], "content": content}

    if action_key == "scroll":
        elem = obj_norm.get("element")
        if isinstance(elem, str) and elem.lower() == "window":
            elem = -1
        try:
            elem = int(elem)  # type: ignore[arg-type]
        except Exception:
            return None
        direction = str(obj_norm.get("content", "")).lower()
        if direction not in {"up", "down"}:
            return None
        return {"action_key": "scroll", "element": elem, "content": direction}

    if action_key in {"wait", "goback", "google"}:
        return {"action_key": action_key}

    if action_key == "answer":
        ans = str(obj_norm.get("answer_content", "")).strip()
        if not ans:
            return None
        return {"action_key": "answer", "answer_content": ans}

    return None


def webvoyager_projection(actions: List[str]) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Project LLM outputs into structured WebVoyager actions.

    Input:
        actions: list of model text outputs, each should contain one <action>...</action>
                block (and ideally a <think>...</think> block).
                Inside <action>, prefer a JSON object with keys such as
                {"action_key": "click", "element": 3} etc.
                As a fallback, concise command formats are accepted, e.g.:
                "Click [12]", "Type [3]; [hello]", "Scroll [WINDOW]; [down]",
                "Wait", "GoBack", "Google", "ANSWER; [final answer]".

    Output:
        - normalized_actions: list of dicts matching WebVoyagerEnv.step expectations
        - valids: list[int] flags (1 valid, 0 invalid)
    """

    normalized_actions: List[Dict[str, Any]] = []
    valids: List[int] = [1] * len(actions)

    for i, raw in enumerate(actions):
        original = raw
        # Extract <action> payload
        payload = _extract_action_block(raw)
        if not payload:
            valids[i] = 0
            normalized_actions.append({"action_key": "wait"})
            continue

        # Try JSON first
        obj = _parse_json_like(payload)
        norm = _normalize_action(obj) if obj is not None else None

        # Fallback to concise formats
        if norm is None:
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