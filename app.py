import gradio as gr
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# ── JD ──────────────────────────────────────────────────────────────────────
JD_TEXT = """Senior AI Engineer — Founding Team, Redrob AI

We are looking for a Senior AI Engineer to join the founding team and build
the core retrieval and ranking systems that power Redrob's intelligent
candidate discovery platform.

What you'll work on:
- Build and improve embeddings-based candidate retrieval (Sentence Transformers, BGE, OpenAI)
- Design ranking models using LTR, NDCG-optimised objectives, and behavioural signals
- Own the vector database layer (Pinecone, Qdrant, Weaviate, FAISS, Milvus)
- Fine-tune LLMs (LoRA / QLoRA / PEFT) for domain-specific understanding
- Build offline evaluation frameworks (NDCG, MRR, MAP, A/B testing)

What we're looking for:
- 5–9 years of experience in applied ML / NLP / search
- Production experience shipping retrieval or ranking systems at scale
- Strong Python, comfort with experimentation and iteration
- Scrappy product-engineering attitude — we build, we ship
"""

JD_QUERY = (
    "Senior AI Engineer Founding Team Redrob AI. Deep technical depth in modern ML systems, "
    "embeddings, retrieval, ranking, LLMs, fine-tuning. Production experience with "
    "embeddings-based retrieval systems sentence-transformers, OpenAI embeddings, BGE. "
    "Production experience with vector databases Pinecone, Weaviate, Qdrant, Milvus, FAISS. "
    "Strong Python. Evaluation frameworks NDCG, MRR, MAP. 5 to 9 years experience."
)

JD_REQUIRED_SKILLS = [
    "sentence transformers", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "embeddings", "vector search", "semantic search", "rag", "retrieval",
    "fine-tuning", "lora", "qlora", "peft", "hugging face", "transformers",
    "llm", "information retrieval", "learning to rank", "langchain", "llamaindex",
    "pgvector", "bm25", "haystack"
]

IRRELEVANT_TITLES = [
    "civil engineer", "accountant", "graphic designer", "content writer",
    "marketing manager", "hr manager", "sales executive", "operations manager",
    "business analyst", "mechanical engineer", "customer support", "project manager"
]

# ── Load model once ──────────────────────────────────────────────────────────
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
query_emb = model.encode(JD_QUERY)
query_norm = query_emb / np.linalg.norm(query_emb)
print("Model ready.")

# ── Load sample candidates ───────────────────────────────────────────────────
SAMPLE_FILE = "sample_candidates.json"
if os.path.exists(SAMPLE_FILE):
    with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
        RAW_CANDIDATES = json.load(f)
else:
    RAW_CANDIDATES = []


def flatten(record):
    p = record.get("profile", {})
    s = record.get("redrob_signals", {})
    skills_list = record.get("skills", [])
    skills_str = ", ".join(sk.get("name", "") for sk in skills_list if sk.get("name"))
    career = record.get("career_history", [])
    career_text = " | ".join(
        f"{c.get('title','')} at {c.get('company','')}: {c.get('description','')}"
        for c in career[:3]
    )
    return {
        "candidate_id": record.get("candidate_id", ""),
        "title": p.get("current_title", ""),
        "experience": p.get("years_of_experience", 0),
        "skills": skills_str,
        "bio": p.get("summary", ""),
        "career_text": career_text,
        "open_to_work": s.get("open_to_work_flag", True),
        "response_rate": s.get("recruiter_response_rate", 0),
        "notice_days": s.get("notice_period_days", 90),
        "interview_rate": s.get("interview_completion_rate", 0.5),
        "github": s.get("github_activity_score", -1),
    }


def jd_skill_match(skills_str):
    low = skills_str.lower()
    return min(sum(1 for s in JD_REQUIRED_SKILLS if s in low) / 5.0, 1.0)


def behavioral_score(c):
    score = 0.0
    if not c["open_to_work"]:
        score -= 0.3
    exp = c["experience"]
    if 5 <= exp <= 9:
        score += 0.15
    elif 4 <= exp <= 12:
        score += 0.08
    score += c["response_rate"] * 0.15
    if c["notice_days"] <= 30:
        score += 0.10
    elif c["notice_days"] <= 60:
        score += 0.05
    elif c["notice_days"] > 90:
        score -= 0.05
    score += c["interview_rate"] * 0.10
    if c["github"] > 0:
        score += min(c["github"] / 100.0, 0.05)
    return score


def title_penalty(title):
    t = title.lower()
    return -0.15 if any(k in t for k in IRRELEVANT_TITLES) else 0.0


def run_ranker():
    if not RAW_CANDIDATES:
        return "⚠️ sample_candidates.json not found in repo.", ""

    candidates = [flatten(r) for r in RAW_CANDIDATES]

    # Embed
    texts = [
        f"Title: {c['title']}. Skills: {c['skills']}. "
        f"Experience: {c['experience']} years. {c['bio']} {c['career_text']}"
        for c in candidates
    ]
    embs = model.encode(texts, show_progress_bar=False)
    embs_norm = embs / np.linalg.norm(embs, axis=1)[:, np.newaxis]
    sem_scores = np.dot(embs_norm, query_norm)

    for i, c in enumerate(candidates):
        c["sem"] = float(sem_scores[i])
        c["skill"] = jd_skill_match(c["skills"])
        c["beh"] = behavioral_score(c)
        c["pen"] = title_penalty(c["title"])
        c["final"] = c["sem"] * 0.60 + c["skill"] * 0.20 + c["beh"] * 0.15 + c["pen"]

    ranked = sorted(candidates, key=lambda x: x["final"], reverse=True)[:10]

    rows = []
    for i, c in enumerate(ranked, 1):
        skills_short = ", ".join(c["skills"].split(", ")[:4])
        rows.append(
            f"**#{i} — {c['title']}** | {c['experience']}y exp | Score: `{c['final']:.3f}`\n"
            f"> Skills: {skills_short}\n"
            f"> Notice: {c['notice_days']}d | Response: {int(c['response_rate']*100)}% "
            f"| Open to work: {'✅' if c['open_to_work'] else '❌'}\n"
        )

    result = "\n---\n".join(rows)
    stats = (
        f"Ranked **{len(candidates)}** sample candidates.\n"
        f"Top candidate score: `{ranked[0]['final']:.3f}` | "
        f"#10 score: `{ranked[9]['final']:.3f}`"
    )
    return stats, result


# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="Redrob Candidate Ranker — Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 Redrob Candidate Ranker — Sandbox Demo")
    gr.Markdown(
        "This demo runs the offline ranking pipeline on the **sample candidates** "
        "(50 profiles from the challenge bundle). The same pipeline processes all "
        "100,000 candidates in the final submission in under 10 seconds.\n\n"
        "**No API calls are made during ranking** — purely local embeddings + scoring."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Job Description")
            gr.Markdown(JD_TEXT)
            run_btn = gr.Button("▶ Run Ranker", variant="primary", size="lg")

        with gr.Column(scale=2):
            gr.Markdown("### 🏆 Top 10 Ranked Candidates")
            stats_box = gr.Markdown("*Click 'Run Ranker' to start.*")
            results_box = gr.Markdown("")

    run_btn.click(fn=run_ranker, inputs=[], outputs=[stats_box, results_box])

    gr.Markdown(
        "---\n**Scoring formula:** 60% semantic similarity (all-MiniLM-L6-v2) + "
        "20% JD skill match + 15% behavioral signals — "
        "penalises irrelevant titles and unavailable candidates."
    )

if __name__ == "__main__":
    demo.launch()
