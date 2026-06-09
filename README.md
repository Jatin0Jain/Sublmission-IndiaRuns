# Intelligent Candidate Discovery & Ranking — Redrob Hackathon

Ranking system for the [Redrob Hackathon](https://redrob.io) challenge: given 100,000 candidate profiles and a job description for a Senior AI Engineer, surface the top 100 best-fit candidates with a ranked CSV and per-candidate reasoning.

## Architecture

Two-stage offline pipeline:

**Stage 1 — Pre-computation** (run once, ahead of evaluation)
- `prep_data.py` — flattens `candidates.jsonl` into a clean parquet, extracts all 23 behavioral signals, detects honeypot profiles
- `prep_embeddings.py` — encodes all 100k profiles with `all-MiniLM-L6-v2` into a numpy array for fast cosine similarity
- `prep_reasonings.py` — uses Gemini to pre-generate nuanced, fact-specific reasoning strings for the top candidates

**Stage 2 — Ranking** (must run in <5 min, CPU only, no network)
- `generate_submission.py` — loads precomputed artefacts, ranks candidates using a composite score (semantic similarity + JD skill match + behavioral signals), outputs `submission.csv`

**Sandbox / Demo**
- `api.py` + `frontend/` — a FastAPI + React UI for interactive candidate exploration (used as the sandbox demo)

## Usage

```bash
# 1. Pre-compute (one-time, ~15 min on CPU)
python prep_data.py --data-dir ./data
python prep_embeddings.py --data-dir ./data
python prep_reasonings.py --data-dir ./data

# 2. Generate submission (offline, ~10 seconds)
python generate_submission.py --candidates ./data/candidates.jsonl --out submission.csv

# 3. Validate
python validate_submission.py submission.csv
```

## Scoring

| Metric | Weight |
|---|---|
| NDCG@10 | 50% |
| NDCG@50 | 30% |
| MAP | 15% |
| P@10 | 5% |

## Compute constraints met
- Runtime: ~10 seconds (well under 5-minute limit)
- Memory: <4 GB RAM
- No GPU, no network calls during ranking
