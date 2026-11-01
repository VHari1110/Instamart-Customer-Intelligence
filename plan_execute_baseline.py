"""Offline Plan-and-Execute baseline for the Instamart refund agent."""

import json
import tempfile
import time
from pathlib import Path

import mock_llm
import refund_agent as agent


def run_plan_execute(user_message, scenario_id="MANUAL"):
    start_time = time.monotonic()
    llm_calls = 0
    tokens_used = 0
    steps = []
    status = "ERROR"
    final_answer = ""
    refund_amount = None
    original_ledger_path = agent.LEDGER_PATH

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent.LEDGER_PATH = Path(temp_dir) / "test_ledger.jsonl"

            messages = [
                {
                    "role": "system",
                    "content": "PLAN"
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]

            # Phase 1: Create the plan.
            response = mock_llm.chat(messages)
            llm_calls += 1
            tokens_used += response.usage.total_tokens

            plan = json.loads(response.choices[0].message.content)

            if plan["status"] != "READY":
                status = plan["status"]
                final_answer = plan.get("reason", "More information is required.")

            else:
                order_id = plan["order_id"]

                # Phase 2: Execute the planned tools.
                order = agent.get_order(order_id)
                steps.append({
                    "tool": "get_order",
                    "observation": order
                })

                if "error" in order:
                    status = "NOT_FOUND"
                    final_answer = "NOT_FOUND: no such order."

                else:
                    policy = agent.search_policy(
                        "refund policy late delivery"
                    )
                    steps.append({
                        "tool": "search_policy",
                        "observation": policy
                    })

                    if order["status"] != "completed":
                        status = "DENIED"
                        final_answer = (
                            "DENIED: only completed orders can be refunded."
                        )

                    else:
                        calculation = agent.calculate_refund(
                            order["order_amount"],
                            order["delivery_minutes"]
                        )
                        steps.append({
                            "tool": "calculate_refund",
                            "observation": calculation
                        })

                        refund_amount = calculation["refund_inr"]

                        if refund_amount <= 0:
                            status = "DENIED"
                            final_answer = (
                                "DENIED: delivery was within 45 minutes."
                            )

                        elif refund_amount > 1000:
                            action = agent.escalate_to_human(
                                order_id,
                                refund_amount
                            )
                            tool_name = "escalate_to_human"

                        else:
                            action = agent.issue_refund(
                                order_id,
                                refund_amount
                            )
                            tool_name = "issue_refund"

                        if refund_amount > 0:
                            steps.append({
                                "tool": tool_name,
                                "observation": action
                            })

                            if action.get("status") == "issued":
                                status = "REFUND"
                                final_answer = (
                                    f"REFUND: INR {refund_amount:.2f} issued."
                                )
                            elif action.get("status") == "pending_human_approval":
                                status = "ESCALATE"
                                final_answer = (
                                    "ESCALATE: refund sent for human approval."
                                )
                            else:
                                status = "DENIED"
                                final_answer = (
                                    "DENIED: " + action.get(
                                        "message", "Refund action denied."
                                    )
                                )

    except Exception as exc:
        status = "ERROR"
        refund_amount = None
        final_answer = f"ERROR: {exc}"

      
    finally:
        agent.LEDGER_PATH = original_ledger_path

    # Phase 3: Generate the final customer response.
    final_messages = [
        {
            "role": "system",
            "content": "PLAN_FINAL"
        },
        {
            "role": "user_result",
            "content": json.dumps({
                "status": status,
                "refund_inr": refund_amount or 0,
                "message": final_answer
            })
        }
    ]

    response = mock_llm.chat(final_messages)
    llm_calls += 1
    tokens_used += response.usage.total_tokens
    final_answer = response.choices[0].message.content

    return {
        "scenario_id": scenario_id,
        "method": "Plan-and-Execute",
        "status": status,
        "final_answer": final_answer,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "tool_calls": len(steps),
        "steps": steps,
        "latency_seconds": round(time.monotonic() - start_time, 4)
    }


if __name__ == "__main__":
    mock_llm.reset()

    result = run_plan_execute(
        "Process refund for ORD-104043",
        scenario_id="BASELINE_S1"
    )

    print(json.dumps(result, indent=2))