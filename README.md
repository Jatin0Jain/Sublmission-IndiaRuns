# Intelligent Candidate Discovery & Ranking — Redrob Hackathon

Ranking system for the [Redrob Hackathon](https://redrob.io) challenge: given 100,000 candidate profiles and a job description for a Senior AI Engineer, surface the top 100 best-fit candidates with a ranked CSV and per-candidate reasoning.

## Architecture

Two-stage offline pipeline:

**Stage 1 — Pre-computation** (run once, ahead of evaluation)
- `prep_data.py` — flattens `candidates.jsonl` into a clean Parquet, extracts all 23 behavioral signals, detects honeypot profiles using 5 independent heuristics
- `prep_embeddings.py` — encodes all 100k candidate profiles with `all-MiniLM-L6-v2` into a `.npy` array for fast cosine similarity
- `prep_reasonings.py` — uses Gemini 2.5 Flash to pre-generate nuanced, fact-specific reasoning strings for the top candidates; results cached in `candidate_reasonings.json`

**Stage 2 — Ranking** (must run in <5 min, CPU only, no network calls)
- `rank.py` — loads the pre-computed parquet, performs a fast 2-phase rule-based ranking, loads pre-generated reasonings, and outputs `submission.csv`
  - *Phase 1:* Reads `candidates_clean.parquet` to rapidly filter 100k → top 2,000 via title matching and skill scoring
  - *Phase 2:* Streams `candidates.jsonl` for only those 2,000 IDs, computes a deep 6-component score, ranks and outputs top 100

**Sandbox / Demo**
- `app.py` — Gradio app for interactive demo of the full ranking pipeline on the 100k dataset (hosted on HuggingFace Spaces)
- `api.py` + `frontend/` — FastAPI + React UI for advanced candidate exploration with LLM-powered re-ranking (local sandbox only)

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Pre-compute (one-time, ~15 min on CPU)
python prep_data.py --data-dir ./data
python prep_embeddings.py --data-dir ./data
python prep_reasonings.py --data-dir ./data   # optional: enriches reasonings with Gemini

# 2. Generate submission (fully offline, ~10 seconds)
python rank.py --candidates ./data/candidates.jsonl --out ./team_submission.csv

# 3. Validate output
python validate_submission.py team_submission.csv
```

## Ranking Formula

The scorer (`rank.py`) uses a **6-component rule-based score** out of 100:

| Component | Max Points | Description |
|---|---|---|
| Title / Career Match | 30 | Strong title = 30, career evidence for soft titles |
| Skills Quality | 25 | Per-skill relevance × proficiency × duration × endorsements |
| Experience Years | 15 | Sweet spot: 5–9 years = full 15 pts |
| Company Type | 15 | Product co. > consulting firm; pure consulting = 0 |
| Location | 10 | Noida/Pune > Tier-1 India cities > willing to relocate |
| Behavioral Multiplier | ×0.40–1.15 | Activity, response rate, notice period, GitHub activity |

## Hackathon Evaluation Metrics

| Metric | Weight |
|---|---|
| NDCG@10 | 50% |
| NDCG@50 | 30% |
| MAP | 15% |
| P@10 | 5% |

## Compute Constraints Met

- **Runtime:** ~10 seconds end-to-end (well under the 5-minute limit)
- **Memory:** <4 GB RAM peak
- **No GPU** required
- **No network calls** during ranking (fully offline)
- **Honeypot protection:** 5-heuristic detection; zero honeypots in final top 100
