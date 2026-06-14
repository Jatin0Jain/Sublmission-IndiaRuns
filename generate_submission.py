import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import time
import json

# ---------------------------------------------------------------------------
# JD-required skills for keyword-match bonus
# ---------------------------------------------------------------------------
JD_REQUIRED_SKILLS = [
    "sentence transformers", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "openai embeddings", "bge", "embeddings", "vector search", "semantic search",
    "rag", "retrieval", "fine-tuning", "finetuning", "lora", "qlora", "peft",
    "hugging face", "transformers", "llm", "ndcg", "mrr", "a/b testing",
    "information retrieval", "learning to rank", "langchain", "llamaindex",
    "pgvector", "bm25", "haystack"
]

# Titles that are clearly irrelevant to an AI Engineer role
IRRELEVANT_TITLE_KEYWORDS = [
    "civil engineer", "accountant", "graphic designer", "content writer",
    "marketing manager", "hr manager", "sales executive", "operations manager",
    "business analyst", "mechanical engineer", "customer support", "project manager"
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default="candidates.jsonl")
    parser.add_argument("--out", type=str, default="submission.csv")
    return parser.parse_args()


def calculate_behavioral_score(row):
    """Multi-signal behavioral score (0 to 1)."""
    score = 0.0

    # Hard penalty: not open to work
    if not row.get("open_to_work_flag", True):
        score -= 0.3

    # Experience match (JD: 5-9 years ideal)
    exp = row.get("experience_years", 0)
    if 5 <= exp <= 9:
        score += 0.15
    elif 4 <= exp < 5 or 9 < exp <= 12:
        score += 0.08
    elif exp > 12:
        score += 0.04

    # Recruiter response rate (engagement signal)
    resp = row.get("recruiter_response_rate", 0)
    score += resp * 0.15

    # Notice period (<= 30 days preferred)
    np_days = row.get("notice_period_days", 90)
    if np_days <= 30:
        score += 0.10
    elif np_days <= 60:
        score += 0.05
    elif np_days > 90:
        score -= 0.05

    # Interview completion rate (shows up when called)
    icr = row.get("interview_completion_rate", 0.5)
    score += icr * 0.10

    # Offer acceptance rate (not a ghost)
    oar = row.get("offer_acceptance_rate", -1)
    if oar >= 0:
        score += oar * 0.05

    # Activity recency
    try:
        days_inactive = (pd.Timestamp.now() - row["last_active"]).days
    except Exception:
        days_inactive = 60
    if days_inactive < 7:
        score += 0.10
    elif days_inactive < 30:
        score += 0.05

    # Saved by recruiters (social proof)
    saved = row.get("saved_by_recruiters_30d", 0)
    score += min(saved / 20.0, 0.05)

    # GitHub activity (shows coding habit)
    gh = row.get("github_activity_score", -1)
    if gh > 0:
        score += min(gh / 100.0, 0.05)

    # Verification trust signals
    if row.get("verified_email", False):
        score += 0.02
    if row.get("verified_phone", False):
        score += 0.02
    if row.get("linkedin_connected", False):
        score += 0.01

    return score


def calculate_jd_skill_match(skills_str):
    """Count how many JD-required skills the candidate explicitly lists."""
    skills_lower = skills_str.lower()
    matches = sum(1 for s in JD_REQUIRED_SKILLS if s in skills_lower)
    # Normalize: 5+ matches = max bonus
    return min(matches / 5.0, 1.0)


def has_irrelevant_title(title_str):
    """Check if the candidate's current title is clearly unrelated to AI engineering."""
    title_lower = title_str.lower()
    return any(kw in title_lower for kw in IRRELEVANT_TITLE_KEYWORDS)


def load_reasonings(data_dir):
    """Load pre-computed Gemini reasonings from the same directory."""
    for path in [
        os.path.join(data_dir, "candidate_reasonings.json"),
        "candidate_reasonings.json",
    ]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def generate_reasoning(row, reasonings_dict):
    """Fetch pre-computed LLM reasoning or fall back to a fact-based template."""
    cid = row.get("candidate_id")
    if cid in reasonings_dict:
        return reasonings_dict[cid]

    exp = row.get("experience_years", 0)
    title = row.get("current_title", "Engineer")
    resp_rate = int(row.get("recruiter_response_rate", 0) * 100)
    notice = row.get("notice_period_days", 30)
    skills = str(row.get("skills", "")).split(",")[:3]
    skills_str = ", ".join(s.strip() for s in skills)

    return (
        f"{exp} years experience as {title} with skills in {skills_str}. "
        f"Behavioral signals: {resp_rate}% response rate, {notice}-day notice period."
    )


def main():
    start_time = time.time()
    args = parse_args()

    data_dir = os.path.dirname(args.candidates)
    if not data_dir:
        data_dir = "."

    parquet_path = os.path.join(data_dir, "candidates_clean.parquet")
    embeddings_path = os.path.join(data_dir, "candidate_embeddings.npy")

    # Fallback to current dir
    if not os.path.exists(parquet_path):
        parquet_path = "candidates_clean.parquet"
        embeddings_path = "candidate_embeddings.npy"

    print("Loading data...")
    df = pd.read_parquet(parquet_path)
    embeddings = np.load(embeddings_path)

    # JD query
    jd_query = (
        "Senior AI Engineer Founding Team Redrob AI. Deep technical depth in modern ML systems, "
        "embeddings, retrieval, ranking, LLMs, fine-tuning. Production experience with "
        "embeddings-based retrieval systems sentence-transformers, OpenAI embeddings, BGE. "
        "Production experience with vector databases Pinecone, Weaviate, Qdrant, Milvus, FAISS. "
        "Strong Python code quality. Hands-on experience designing evaluation frameworks NDCG, MRR, "
        "MAP, offline-to-online correlation, A/B testing. 5 to 9 years experience. "
        "Scrappy product-engineering attitude."
    )

    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(jd_query)

    print("Calculating similarities...")
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    similarities = np.dot(embeddings_norm, query_norm)
    df["semantic_score"] = similarities

    print("Applying filters and scoring...")
    # Filter honeypots
    df = df[df["is_honeypot"] == 0].copy()

    # Hard filter: not open to work gets heavy penalty already in behavioral score
    # We keep them in but their score will be depressed

    # JD skill match bonus
    df["skill_match_score"] = df["skills"].apply(calculate_jd_skill_match)

    # Title relevance penalty
    df["title_penalty"] = df["current_title"].apply(
        lambda t: -0.15 if has_irrelevant_title(str(t)) else 0.0
    )

    # Behavioral score
    df["behavioral_score"] = df.apply(calculate_behavioral_score, axis=1)

    # Composite score:
    # 60% semantic (JD text match) + 20% skill keyword match + 15% behavioral + title penalty
    df["final_score"] = (
        df["semantic_score"] * 0.60
        + df["skill_match_score"] * 0.20
        + df["behavioral_score"] * 0.15
        + df["title_penalty"]
    )

    # Sort and take top 100
    top_100 = df.sort_values(by="final_score", ascending=False).head(100).copy()
    top_100["rank"] = range(1, 101)

    print("Loading reasonings...")
    reasonings_dict = load_reasonings(data_dir)

    print("Generating reasoning...")
    top_100["reasoning"] = top_100.apply(
        lambda row: generate_reasoning(row, reasonings_dict), axis=1
    )

    output_df = top_100[["candidate_id", "rank", "final_score", "reasoning"]].copy()
    output_df.rename(columns={"final_score": "score"}, inplace=True)

    print(f"Writing to {args.out}...")
    output_df.to_csv(args.out, index=False, encoding="utf-8")

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.2f} seconds.")
    print(f"Top 10 candidates: {list(top_100['candidate_id'].head(10))}")


if __name__ == "__main__":
    main()
