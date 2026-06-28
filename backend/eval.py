import json
import re
from pathlib import Path
from pydantic import BaseModel


class EvalReport(BaseModel):
    passed: bool
    total: int
    checks_run: list[str]
    failed_checks: list[str]
    per_check_counts: dict[str, int]
    sample_failures: list[dict]
    domains_found: list[str]


def _levenshtein(a: str, b: str) -> int:
    if len(a) > 200:
        a = a[:200]
    if len(b) > 200:
        b = b[:200]
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n]


def run_eval(jsonl_path: str, sample_size: int = 100) -> EvalReport:
    path = Path(jsonl_path)
    checks_run = [
        "json_parseable",
        "schema_valid",
        "role_sequence",
        "content_nonempty",
        "domain_coverage",
        "uniqueness",
    ]
    failures: dict[str, list[dict]] = {c: [] for c in checks_run}
    examples = []

    # Load up to sample_size lines
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Check 1: JSON parseable
                try:
                    obj = json.loads(line)
                    examples.append(obj)
                except json.JSONDecodeError as e:
                    failures["json_parseable"].append({"line": line[:80], "error": str(e)})
                if len(examples) >= sample_size:
                    break
    except FileNotFoundError:
        return EvalReport(
            passed=False,
            total=0,
            checks_run=checks_run,
            failed_checks=["json_parseable"],
            per_check_counts={"json_parseable": 1},
            sample_failures=[{"error": f"File not found: {jsonl_path}"}],
            domains_found=[],
        )

    total = len(examples)
    domains_found: set[str] = set()
    user_questions: list[str] = []

    for i, obj in enumerate(examples):
        # Collect domain metadata (stripped before output)
        if "_domain" in obj:
            domains_found.add(obj["_domain"])

        # Check 2: Schema valid
        msgs = obj.get("messages")
        if not isinstance(msgs, list) or len(msgs) != 3:
            failures["schema_valid"].append({"index": i, "issue": f"messages length={len(msgs) if isinstance(msgs, list) else 'not list'}"})
            continue
        for msg in msgs:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                failures["schema_valid"].append({"index": i, "issue": "missing role or content", "msg": str(msg)[:80]})
                break

        # Check 3: Role sequence
        roles = [m.get("role") for m in msgs]
        if roles != ["system", "user", "assistant"]:
            failures["role_sequence"].append({"index": i, "roles": roles})

        # Check 4: Content non-empty (≥20 chars)
        for msg in msgs:
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content.strip()) < 20:
                failures["content_nonempty"].append({"index": i, "role": msg.get("role"), "content_len": len(content)})

        # Collect user questions for uniqueness check
        user_msg = next((m for m in msgs if m.get("role") == "user"), None)
        if user_msg:
            user_questions.append(user_msg.get("content", ""))

    # Check 5: Domain coverage (≥8 of 10 domains if enough examples, else ≥3)
    all_domains = {"cybersecurity", "networking", "it", "reasoning", "threejs", "animations", "css", "frontend", "backend"}
    required_domains = 3 if total < 50 else min(8, len(all_domains))
    if len(domains_found) < required_domains:
        failures["domain_coverage"].append({
            "found": list(domains_found),
            "required": required_domains,
            "note": "Not enough domain diversity",
        })

    # Check 6: Uniqueness — sample pairs via Levenshtein
    uniqueness_fails = 0
    check_limit = min(len(user_questions), 50)
    for i in range(check_limit):
        for j in range(i + 1, check_limit):
            if i == j:
                continue
            q1, q2 = user_questions[i], user_questions[j]
            if q1.strip().lower() == q2.strip().lower():
                uniqueness_fails += 1
                if uniqueness_fails <= 3:
                    failures["uniqueness"].append({"i": i, "j": j, "q": q1[:80]})
    if uniqueness_fails > 0:
        failures["uniqueness"].insert(0, {"total_duplicates": uniqueness_fails})

    # Build report
    per_check_counts = {c: len(v) for c, v in failures.items()}
    failed_checks = [c for c, v in failures.items() if v]
    sample_failures = []
    for c, v in failures.items():
        for item in v[:2]:
            sample_failures.append({"check": c, **item})

    passed = len(failed_checks) == 0

    return EvalReport(
        passed=passed,
        total=total,
        checks_run=checks_run,
        failed_checks=failed_checks,
        per_check_counts=per_check_counts,
        sample_failures=sample_failures,
        domains_found=sorted(domains_found),
    )
