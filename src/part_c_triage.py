"""Part C: support-ticket triage with guardrails (redaction, retries, validation, budget, routing)."""
import json, logging, os, random, re, sys, time
sys.path.insert(0, os.getcwd())
import pandas as pd
import mock_llm
from mock_llm import TransientLLMError, est_tokens

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
CATEGORIES = ["billing", "delivery", "technical", "fraud"]

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("triage")
random.seed(42)                                       # makes jitter reproducible

# ------------------------------------------------------------------ redaction (exact patterns, exact order)
PATTERNS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]

def redact(text):
    counts = {}
    for name, pat, ph in PATTERNS:
        text, n = pat.subn(ph, text)
        counts[name] = n
    return text, counts

# ------------------------------------------------------------------ prompts
FORMAT = ('Reply with JSON only, in the form {"category": "...", "reason": "..."}. '
          'Allowed categories: billing, delivery, technical, fraud.')
ZERO_SHOT = "You CLASSIFY customer support tickets. " + FORMAT
FEW_SHOT = ("You CLASSIFY customer support tickets. " + FORMAT + "\nEXAMPLES:\n"
            "Ticket: Someone hacked my account and used my wallet. => "
            '{"category": "fraud", "reason": "unauthorized account use"}\n'
            "Ticket: The app freezes when I apply a coupon. => "
            '{"category": "technical", "reason": "app error"}\n'
            "Ticket: I was charged twice for [ORDER_ID]. => "
            '{"category": "billing", "reason": "double charge"}\n'
            "Ticket: The rider has not arrived and is very late. => "
            '{"category": "delivery", "reason": "late delivery"}')

# ------------------------------------------------------------------ resilient call
retry_log = []
transient_errors = 0

def call_with_retry(messages, ticket_id):
    """Up to MAX_RETRIES retries; wait before retry n = BASE_DELAY * 2**n * uniform(0.9, 1.15)."""
    global transient_errors
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            transient_errors += 1
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            retry_log.append((ticket_id, n, wait))
            log.info(f"RETRY ticket={ticket_id} retry={n} wait={wait:.4f}s")
            time.sleep(wait)

# ------------------------------------------------------------------ output validation
def validate(content):
    """Returns (status, category). Never guesses."""
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and obj.get("category") in CATEGORIES:
            return "classified_json", obj["category"]
        return "flagged_invalid_output", ""
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", (content or "").lower())
        found = {w for w in words if w in CATEGORIES}
        if len(found) == 1:
            return "classified_keyword", next(iter(found))
        return "flagged_invalid_output", ""

# ------------------------------------------------------------------ pipeline
def run_pipeline(tickets, system_prompt, token_budget):
    global transient_errors
    mock_llm.reset()
    transient_errors = 0
    retry_log.clear()
    total_tokens, out = 0, []
    for t in tickets.itertuples():
        text = t.text
        rec = dict(ticket_id=t.ticket_id, customer_id=t.customer_id, status="", category="", tokens_used=0)
        if text.strip() == "":
            rec["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            rec["status"] = "flagged_long"
        elif total_tokens >= token_budget:
            rec["status"] = "budget_exhausted"
        else:
            clean, _ = redact(text)
            msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": clean}]
            assert msgs[1]["content"] == clean
            for _, pat, _ in PATTERNS:                       # no PII may reach the model
                assert not pat.search(msgs[1]["content"]), "PII leaked into prompt"
            resp = call_with_retry(msgs, t.ticket_id)
            used = resp.usage.total_tokens
            total_tokens += used
            rec["tokens_used"] = used
            rec["status"], rec["category"] = validate(resp.choices[0].message.content)
        out.append(rec)
    return pd.DataFrame(out), total_tokens, transient_errors

def route(row, premium):
    if row["status"] in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
        return "human_review", "NORMAL"
    if row["category"] == "fraud":
        return "security_team", "HIGH"
    if premium:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"

if __name__ == "__main__":
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    print(f"tickets: {len(tickets)}")

    # --- task 1: guardrails
    n_empty = int((tickets["text"].str.strip() == "").sum())
    long_mask = tickets["text"].map(est_tokens) > MAX_INPUT_TOKENS
    print(f"empty tickets: {n_empty} | long tickets: {int(long_mask.sum())} "
          f"(estimated tokens {sorted(set(tickets.loc[long_mask, 'text'].map(est_tokens)))})")

    # --- task 2: redaction counts + verification
    tot = {"email": 0, "phone": 0, "order_id": 0}
    redacted = []
    for txt in tickets["text"]:
        r, c = redact(txt)
        redacted.append(r)
        for k in tot:
            tot[k] += c[k]
    left = sum(len(p.findall(r)) for r in redacted for _, p, _ in PATTERNS)
    print(f"substitutions: {tot} total {sum(tot.values())} | matches left after redaction: {left}")
    assert left == 0

    # --- task 3
    print(f"zero-shot prompt est tokens: {est_tokens(ZERO_SHOT)} | few-shot: {est_tokens(FEW_SHOT)}")
    assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT

    # --- tasks 4-6: full runs, no budget
    summary = {}
    for label, prompt in [("zero-shot", ZERO_SHOT), ("few-shot", FEW_SHOT)]:
        print(f"\n===== full run: {label} =====")
        df, tok, nerr = run_pipeline(tickets, prompt, token_budget=10 ** 12)
        df = df.merge(tickets[["ticket_id", "true_category"]], on="ticket_id")
        cls = df[df["status"].isin(["classified_json", "classified_keyword"])]
        acc = (cls["category"] == cls["true_category"]).mean()
        waits = [w for _, _, w in retry_log]
        ok_range = all(BASE_DELAY * 2 ** n * 0.9 <= w <= BASE_DELAY * 2 ** n * 1.15 for _, n, w in retry_log)
        print(f"status counts: {df['status'].value_counts().to_dict()}")
        print(f"TransientLLMErrors: {nerr} | retries logged: {len(retry_log)} | all waits in range: {ok_range} "
              f"| wait min {min(waits):.4f} max {max(waits):.4f}")
        assert nerr == len(retry_log) == 19 and ok_range
        print(f"classified {len(cls)} | accuracy {acc*100:.1f}% ({int((cls['category']==cls['true_category']).sum())}/{len(cls)}) "
              f"| total tokens {tok} | tokens/classified {tok/len(cls):.1f}")
        summary[label] = (df, cls)
    zs_cls = summary["zero-shot"][1]
    print("\nZero-shot confusion table (rows = true, columns = predicted):")
    print(pd.crosstab(zs_cls["true_category"], zs_cls["category"]).to_string())

    # --- task 7a: budget demo
    print("\n===== budget demo: zero-shot, TOKEN_BUDGET = 6000 =====")
    df, tok, nerr = run_pipeline(tickets, ZERO_SHOT, token_budget=6000)
    print(f"status counts: {df['status'].value_counts().to_dict()} | tokens used {tok}")
    assert (df["status"] == "budget_exhausted").sum() >= 1
    seg = pd.read_csv("outputs/customer_segments_assignment.csv")
    seg_map = seg.set_index("customer_id")

    def finish(df):
        df = df.copy()
        df["segment_name"] = df["customer_id"].map(seg_map["segment_name"]).fillna("")
        prem = df["customer_id"].map(seg_map["is_premium"]).fillna(False).astype(bool)
        rt = [route(r, p) for (_, r), p in zip(df.iterrows(), prem)]
        df["route"] = [a for a, _ in rt]
        df["priority"] = [b for _, b in rt]
        return df[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]]

    finish(df).to_csv("outputs/triage_budget_demo.csv", index=False)

    # --- task 7b: final results
    print("\n===== final run: few-shot, TOKEN_BUDGET = 100000 =====")
    df, tok, nerr = run_pipeline(tickets, FEW_SHOT, token_budget=100000)
    final = finish(df)
    final.to_csv("outputs/triage_results.csv", index=False)
    print(f"rows {len(final)} | status counts {final['status'].value_counts().to_dict()} | tokens {tok}")
    print("route counts:", final["route"].value_counts().to_dict())
    print("priority counts:", final["priority"].value_counts().to_dict())
    print("rule-validated (classified_json + classified_keyword):",
          int(final["status"].isin(["classified_json", "classified_keyword"]).sum()),
          "| sent to human review:", int((final["route"] == "human_review").sum()))

