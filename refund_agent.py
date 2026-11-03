from pathlib import Path
import sqlite3
import json
from datetime import datetime, timezone

import mock_llm
from policy_rag import load_policy_chunks, build_retriever


DB_PATH = Path("data/instamart.db")
POLICY_DIR = Path("data/policies")
OUTPUT_DIR = Path("outputs")
STATE_DIR = Path("state")

OUTPUT_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)

LEDGER_PATH = OUTPUT_DIR / "refund_ledger.jsonl"


def get_order(order_id):
    """Read one order from the SQLite database."""
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()

    if row is None:
        return {"error": "order not found"}

    return {
        "order_id": row["order_id"],
        "status": row["status"],
        "order_amount": float(row["order_amount"]),
        "delivery_minutes": (
            None if row["delivery_minutes"] is None
            else float(row["delivery_minutes"])
        ),
    }


def search_policy(query):
    """Retrieve the three most relevant policy chunks."""
    chunks = load_policy_chunks()
    retrieve = build_retriever(chunks)
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    """Calculate the refund according to the 45-minute policy."""
    if delivery_minutes is None or float(delivery_minutes) <= 45:
        return {"refund_inr": 0.0}

    return {"refund_inr": round(float(order_amount), 2)}


def write_ledger(tool_name, order_id, refund_inr):
    """Append a simulated refund action to the JSONL ledger."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
    }

    with LEDGER_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")

    return {"refund_inr": round(float(refund_inr), 2)}

def issue_refund(order_id, refund_inr):
    """Record an eligible automatic refund only once per order."""

    order = get_order(order_id)

    if "error" in order:
        return {"status": "denied", "message": "Order not found."}

    if order["status"] != "completed":
        return {"status": "denied", "message": "Only completed orders are eligible."}

    if (
        order["delivery_minutes"] is None
        or order["delivery_minutes"] <= 45
    ):
        return {"status": "denied", "message": "Delivery does not qualify for a late-delivery refund."}

    expected_refund = round(order["order_amount"], 2)
    requested_refund = round(float(refund_inr), 2)

    if requested_refund != expected_refund or requested_refund <= 0:
        return {"status": "denied", "message": "Refund amount does not match the eligible order amount."}

    if requested_refund > 1000:
        return {
            "status": "denied",
            "message": "Refunds above INR 1000 require human approval."
        }

    if LEDGER_PATH.exists():
        with LEDGER_PATH.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                entry = json.loads(line)

                if (
                    entry.get("tool") in ("issue_refund", "escalate_to_human")
                    and entry.get("order_id") == order_id
                ):
                    return {
                        "refund_inr": entry["refund_inr"],
                        "status": "already_processed",
                        "message": "A refund action is already recorded for this order."
                    }

    result = write_ledger("issue_refund", order_id, requested_refund)
    result["status"] = "issued"
    return result


def escalate_to_human(order_id, refund_inr):
    """Record an eligible refund requiring human approval."""

    order = get_order(order_id)

    if "error" in order:
        return {"status": "denied", "message": "Order not found."}

    if order["status"] != "completed":
        return {"status": "denied", "message": "Only completed orders are eligible."}

    if (
        order["delivery_minutes"] is None
        or order["delivery_minutes"] <= 45
    ):
        return {"status": "denied", "message": "Delivery does not qualify for a late-delivery refund."}

    expected_refund = round(order["order_amount"], 2)
    requested_refund = round(float(refund_inr), 2)

    if requested_refund != expected_refund or requested_refund <= 1000:
        return {
            "status": "denied",
            "message": "Human escalation is required only for eligible refunds above INR 1000."
        }

    if LEDGER_PATH.exists():
        with LEDGER_PATH.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                entry = json.loads(line)

                if (
                    entry.get("tool") in ("issue_refund", "escalate_to_human")
                    and entry.get("order_id") == order_id
                ):
                    return {
                        "refund_inr": entry["refund_inr"],
                        "status": "already_processed",
                        "message": "A refund action is already recorded for this order."
                    }

    result = write_ledger("escalate_to_human", order_id, requested_refund)
    result["status"] = "pending_human_approval"
    return result




TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}

TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up an order in the SQLite database.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "search_policy",
        "description": "Search the refund and delivery policies.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate_refund",
        "description": "Calculate the eligible refund amount.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"},
            },
            "required": ["order_amount", "delivery_minutes"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Record an eligible refund.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Send a high-value refund for human review.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
]
def run_agent(
    user_message,
    scenario_id="MANUAL",
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tool_count=None,
    resume_data=None,
):
    import time

    run_id = run_id or scenario_id
    start_time = time.monotonic()
    llm_calls = 0
    if resume_data:
        messages = resume_data["messages"]
        steps = resume_data["steps"]
        tokens_used = resume_data["tokens_used"]
        tool_count = len(steps)
    else:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an Instamart refund support AGENT. "
                    "Use tools to check orders, retrieve policy, calculate refunds, "
                    "and issue eligible refunds or escalate refunds above INR 1000. "
                    "Only completed orders may be refunded. Never invent order data."
                ),
            },
            {"role": "user", "content": user_message},
        ]
        steps = []
        tokens_used = 0
        tool_count = 0

    seen_tool_calls = {
        (step["tool"], json.dumps(step["arguments"], sort_keys=True))
        for step in steps
    }

    status = None
    final_answer = ""

    for iteration in range(max_iterations):
        if time.monotonic() - start_time >= time_limit_seconds:
            status = "FALLBACK_TIMEOUT"
            final_answer = "Human handoff required: time limit reached."
            break

        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        message = response.choices[0].message

              
        tokens_used += getattr(response.usage, "total_tokens", 0)

        if tokens_used > token_limit:
            status = "FALLBACK_TOKEN_LIMIT"
            final_answer = "Human handoff required: token limit reached."
            break

        tool_calls = getattr(message, "tool_calls", None)

        if not tool_calls:
            final_answer = message.content or ""
            status = final_answer.split(":", 1)[0].strip().upper()
            break

        assistant_calls = []
        for call in tool_calls:
            assistant_calls.append({
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                },
            })

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_calls,
        })

        for call in tool_calls:
            tool_name = call.name
            arguments = call.arguments

            call_key = (
                tool_name,
                json.dumps(arguments, sort_keys=True)
            )

            is_duplicate = call_key in seen_tool_calls

            if is_duplicate:
                observation = {"error": "Repeated tool call blocked."}
            else:
                seen_tool_calls.add(call_key)

                if tool_name not in TOOL_MAP:
                    observation = {"error": f"Unknown tool: {tool_name}"}
                else:
                    try:
                        observation = TOOL_MAP[tool_name](**arguments)
                    except Exception as exc:
                        observation = {"error": str(exc)}

                steps.append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "observation": observation,
                })
                tool_count += 1

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": tool_name,
                "content": json.dumps(observation, default=str),
            })

            if not is_duplicate:
                save_checkpoint(
                    run_id,
                    scenario_id,
                    messages,
                    tokens_used,
                    tool_count,
                    steps,
                )

                if (
                    crash_after_tool_count is not None
                    and tool_count == crash_after_tool_count
                ):
                    raise RuntimeError(
                        f"Simulated crash after tool execution {tool_count}"
                    )
    else:
        status = "FALLBACK_MAX_ITERATIONS"
        final_answer = "Human handoff required: maximum iterations reached."

    if status is None:
        status = "FALLBACK_MAX_ITERATIONS"
        final_answer = "Human handoff required: maximum iterations reached."

    result = {
        "scenario_id": scenario_id,
        "run_id": run_id,
        "status": status,
        "final_answer": final_answer,
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
    }

    return result
def save_checkpoint(run_id, scenario_id, messages, tokens_used, step_number, steps):
    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step_number": step_number,
        "steps": steps,
    }

    path = STATE_DIR / f"{run_id}.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(checkpoint, file, indent=2, default=str)

def resume_agent(run_id):
    path = STATE_DIR / f"{run_id}.json"

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        checkpoint = json.load(file)

    result = run_agent(
        user_message=next(
            message["content"]
            for message in checkpoint["messages"]
            if message["role"] == "user"
        ),
        scenario_id=checkpoint["scenario_id"],
        run_id=run_id,
        resume_data=checkpoint,
    )

    return result


def run_all_scenarios():
    import csv

    scenarios_path = Path("data/raw/agent_scenarios.csv")
    runs_path = OUTPUT_DIR / "agent_runs.jsonl"
    results = []

    with scenarios_path.open("r", encoding="utf-8-sig", newline="") as file:
        scenarios = list(csv.DictReader(file))

    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        message = scenario["message"]

        result = run_agent(
            user_message=message,
            scenario_id=scenario_id,
            run_id=f"RUN_{scenario_id}",
            max_iterations=6,
        )

        log_entry = {
            "scenario_id": scenario_id,
            "status": result["status"],
            "final_answer": result["final_answer"],
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "steps": result["steps"],
        }

        with runs_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(log_entry, default=str) + "\n")

        results.append(result)
        print(
            f"{scenario_id}: {result['status']} | "
            f"{result['final_answer']}"
        )

    return results
