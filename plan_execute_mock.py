"""Offline mock LLM dedicated to the Plan-and-Execute baseline."""

import json
import math
import re
from types import SimpleNamespace as NS


def est_tokens(text):
    return math.ceil(len((text or "").split()) * 1.3)


def _response(content, messages):
    prompt_text = " ".join(
        str(message.get("content", ""))
        for message in messages
    )
    prompt_tokens = est_tokens(prompt_text)
    completion_tokens = est_tokens(content)

    message = NS(
        role="assistant",
        content=content,
        tool_calls=[]
    )

    return NS(
        choices=[NS(message=message, finish_reason="stop")],
        usage=NS(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )


def _create_plan(messages):
    user_message = next(
        (m.get("content", "") for m in messages if m.get("role") == "user"),
        ""
    )

    match = re.search(r"\bORD-\d{6}\b", user_message)

    if not match:
        plan = {
            "status": "NEEDS_INFO",
            "steps": [],
            "reason": "Please provide a valid order ID."
        }
    else:
        plan = {
            "status": "READY",
            "order_id": match.group(0),
            "steps": [
                "get_order",
                "search_policy",
                "calculate_refund",
                "choose_refund_action",
                "return_result"
            ]
        }

    return _response(json.dumps(plan), messages)


def _create_final_response(messages):
    result_text = next(
        (
            m.get("content", "{}")
            for m in reversed(messages)
            if m.get("role") == "user_result"
        ),
        "{}"
    )

    data = json.loads(result_text)
    status = data.get("status", "UNKNOWN")
    amount = float(data.get("refund_inr", 0) or 0)

    if status in ("REFUND", "issued"):
        answer = f"REFUND: INR {amount:.2f} issued."
    elif status in ("ESCALATE", "pending_human_approval"):
        answer = (
            f"ESCALATE: INR {amount:.2f} sent for human approval."
        )
    else:
        answer = data.get(
            "message",
            "The request could not be processed."
        )

    return _response(answer, messages)


def chat(messages, tools=None):
    if not messages:
        raise ValueError("Plan mock requires a system message.")

    system_message = str(messages[0].get("content", "")).upper()

    if "PLAN_FINAL" in system_message:
        return _create_final_response(messages)

    if "PLAN" in system_message:
        return _create_plan(messages)

    raise ValueError(
        "plan_execute_mock supports only PLAN and PLAN_FINAL modes."
    )


def reset():
    """Compatibility function; this deterministic mock has no state."""
    pass
