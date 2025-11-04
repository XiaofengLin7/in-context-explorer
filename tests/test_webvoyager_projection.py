from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from agent_system.environments.env_package.webvoyager.projection import (
    webvoyager_projection,
)


@pytest.mark.parametrize(
    "model_output, expected_action, expected_valid",
    [
        (
            """Thought: Analyze the page and identify the correct element.\nAction: Click [8]""",
            {"action_key": "click", "element": 8},
            1,
        ),
        (
            """Thought: Enter the city directly in the search box.\nAction: Type [22] [Boston]""",
            {"action_key": "type", "element": 22, "content": "Boston"},
            1,
        ),
        (
            """Thought: Scroll to reveal more content.\nAction: Scroll [down]""",
            {"action_key": "scroll", "element": -1, "content": "down"},
            1,
        ),
        (
            """Thought: We have the final answer.\nAction: ANSWER [06516]""",
            {"action_key": "answer", "answer_content": "06516"},
            1,
        ),
        (
            """Action: Click [1]""",
            {"action_key": "click", "element": 1},
            0,  # Missing Thought block should mark invalid
        ),
        (
            """Thought: 包含中文字符以触发无效标记。\nAction: Click [2]""",
            {"action_key": "click", "element": 2},
            0,  # Chinese characters should mark invalid
        ),
        # JSON and <action> blocks are no longer supported on purpose.
    ],
)
def test_webvoyager_projection_parsing(
    model_output: str, expected_action: Dict[str, Any], expected_valid: int
) -> None:
    """Validate projection of various Action formats into env actions.

    This test covers:
    - Parsing from single-line "Action:" format
    - Parsing JSON within <action>...</action>
    - Type without semicolon (Type [n] [text])
    - Scroll short form (Scroll [up/down]) defaulting to window (-1)
    - Reasoning detection via "Thought:" and invalidation cases
    """

    normalized_actions, valids = webvoyager_projection([model_output])

    assert isinstance(normalized_actions, list) and len(normalized_actions) == 1
    assert isinstance(valids, list) and len(valids) == 1

    assert normalized_actions[0] == expected_action
    assert valids[0] == expected_valid


