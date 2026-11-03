
import csv
import mock_llm
from refund_agent import run_agent

CSV_PATH = "data/raw/agent_scenarios.csv"

def evaluate():
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        scenarios = list(csv.DictReader(f))

    passed = 0
    failed = 0

    print("\n========== REFUND AGENT EVALUATION ==========\n")

    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        expected_status = scenario["expected_outcome"]
        expected_refund = float(scenario["expected_refund_inr"])

        mock_llm.reset()

        result = run_agent(
            scenario["message"],
            scenario_id=scenario_id,
            run_id=f"EVAL_{scenario_id}"
        )

        actual_status = result["status"]
        steps = result["steps"]

        actual_refund = 0.0
        action_status = "NO_ACTION"

        for step in steps:
            observation = step.get("observation", {})

            if step["tool"] == "calculate_refund":
                actual_refund = float(
                    observation.get("refund_inr", 0)
                )

            if step["tool"] in (
                "issue_refund",
                "escalate_to_human"
            ):
                action_status = observation.get(
                    "status", "UNKNOWN"
                )

        if expected_status == "FALLBACK":
            status_ok = actual_status.startswith("FALLBACK")
        elif expected_status == "REFUND":
            status_ok = actual_status in (
                "REFUND",
                "REFUND_ALREADY_PROCESSED"
            )
        elif expected_status == "ESCALATE":
            status_ok = actual_status in (
                "ESCALATE",
                "ESCALATE_ALREADY_PROCESSED"
            )
        else:
            status_ok = actual_status == expected_status

        refund_ok = abs(actual_refund - expected_refund) < 0.01

        if expected_status == "REFUND":
            action_ok = action_status in (
                "issued",
                "already_processed"
            )
        elif expected_status == "ESCALATE":
            action_ok = action_status in (
                "pending_human_approval",
                "already_processed"
            )
        else:
            action_ok = action_status == "NO_ACTION"

        test_passed = status_ok and refund_ok and action_ok

        if test_passed:
            passed += 1
            label = "PASS"
        else:
            failed += 1
            label = "FAIL"

        print(
            f"{scenario_id}: {label} | "
            f"Expected={expected_status}, Actual={actual_status} | "
            f"Refund={actual_refund:.2f} | "
            f"Action={action_status}"
        )

    total = passed + failed

    print("\n============== FINAL SUMMARY ==============")
    print(f"Total scenarios : {total}")
    print(f"Passed          : {passed}")
    print(f"Failed          : {failed}")
    print(f"Pass rate       : {passed / total * 100:.1f}%")

    print(
        "\nRESULT: ALL SCENARIOS PASSED"
        if failed == 0
        else "\nRESULT: SOME SCENARIOS FAILED"
    )

if __name__ == "__main__":
    evaluate()
