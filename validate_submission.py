"""
validate_submission.py
----------------------
Validates a team_submission.csv against the hackathon rules:
  - Must have exactly 100 rows
  - Required columns: candidate_id, rank, score, reasoning
  - Ranks must be 1–100 with no duplicates or gaps
  - Scores must be in [0, 100]
  - Reasoning must be a non-empty string for every row
  - No duplicate candidate_ids

Usage:
    python validate_submission.py team_submission.csv
"""

import sys
import csv
import os


def validate(path: str) -> bool:
    if not os.path.exists(path):
        print(f"[FAIL] File not found: {path}")
        return False

    required_columns = {"candidate_id", "rank", "score", "reasoning"}
    errors = []
    warnings = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])

        missing = required_columns - columns
        if missing:
            print(f"[FAIL] Missing required columns: {missing}")
            return False

        rows = list(reader)

    # Row count
    if len(rows) != 100:
        errors.append(f"Expected exactly 100 rows, got {len(rows)}")

    seen_ids = set()
    seen_ranks = set()

    for i, row in enumerate(rows, start=2):  # start=2 because row 1 is the header
        cid = row.get("candidate_id", "").strip()
        rank_str = row.get("rank", "").strip()
        score_str = row.get("score", "").strip()
        reasoning = row.get("reasoning", "").strip()

        # candidate_id
        if not cid:
            errors.append(f"Row {i}: empty candidate_id")
        elif cid in seen_ids:
            errors.append(f"Row {i}: duplicate candidate_id '{cid}'")
        else:
            seen_ids.add(cid)

        # rank
        try:
            rank = int(rank_str)
            if rank < 1 or rank > 100:
                errors.append(f"Row {i}: rank {rank} is outside [1, 100]")
            elif rank in seen_ranks:
                errors.append(f"Row {i}: duplicate rank {rank}")
            else:
                seen_ranks.add(rank)
        except ValueError:
            errors.append(f"Row {i}: rank '{rank_str}' is not an integer")

        # score
        try:
            score = float(score_str)
            if score < 0 or score > 100:
                errors.append(f"Row {i}: score {score} is outside [0, 100]")
        except ValueError:
            errors.append(f"Row {i}: score '{score_str}' is not a number")

        # reasoning
        if not reasoning:
            errors.append(f"Row {i}: reasoning is empty")
        elif len(reasoning) < 10:
            warnings.append(f"Row {i}: reasoning is very short ({len(reasoning)} chars)")

    # Check ranks form a complete set 1–100
    if seen_ranks and seen_ranks != set(range(1, len(rows) + 1)):
        missing_ranks = set(range(1, len(rows) + 1)) - seen_ranks
        errors.append(f"Rank gaps detected: {sorted(missing_ranks)}")

    # Report
    if warnings:
        for w in warnings:
            print(f"[WARN] {w}")

    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        print(f"\n[FAIL] Validation FAILED - {len(errors)} error(s) found.")
        return False

    print(f"[PASS] Validation PASSED - {len(rows)} rows, all checks OK.")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_submission.py <path_to_submission.csv>")
        sys.exit(1)

    ok = validate(sys.argv[1])
    sys.exit(0 if ok else 1)
