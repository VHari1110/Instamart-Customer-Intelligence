from pathlib import Path
import refund_agent as ra

ra.OUTPUT_DIR = Path("outputs/regression_fix_20261103")
ra.STATE_DIR = Path("state/regression_fix_20261103")
ra.LEDGER_PATH = ra.OUTPUT_DIR / "refund_ledger.jsonl"

try:
    ra.run_agent(
        "My order ORD-100048 reached me very late. I would like a refund.",
        scenario_id="S1",
        run_id="REGRESSION_FIX_20261103_CRASH_RESUME",
        crash_after_tool_count=1
    )
except RuntimeError as e:
    print("EXPECTED CRASH:", e)

print(
    "CHECKPOINT EXISTS:",
    (ra.STATE_DIR / "REGRESSION_FIX_20261103_CRASH_RESUME.json").exists()
)