import gradio as gr
import json
import os
import math
from datetime import date
from constants import CONSULTING_FIRMS, SKILL_RELEVANCE, PROFICIENCY_MULT, MAX_RAW_SCORE

REFERENCE_DATE = date.today()  # Always relative to when the demo is run

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

SAMPLE_FILE = "sample_candidates.json"
if os.path.exists(SAMPLE_FILE):
    with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
        RAW_CANDIDATES = json.load(f)
else:
    RAW_CANDIDATES = []

# CONSULTING_FIRMS, SKILL_RELEVANCE, PROFICIENCY_MULT imported from constants.py

def score_candidate_full(full_json):
    score = 0.0
    
    # 1. Title + Career Match
    title = full_json.get('profile', {}).get('current_title', '').lower()
    career_titles = [ch.get('title', '').lower() for ch in full_json.get('career_history', [])]
    career_descriptions = ' '.join(ch.get('description', '') for ch in full_json.get('career_history', [])).lower()
    
    
    STRONG_TITLE_KWS = ['ml engineer', 'machine learning', 'ai engineer', 'nlp engineer',
                         'applied scientist', 'search engineer', 'recommendation', 
                         'retrieval', 'ranking', 'applied ml', 'deep learning engineer']
    SOFT_TITLE_KWS = ['data scientist', 'research engineer', 'mlops', 'backend engineer',
                      'software engineer', 'platform engineer']
    
    current_is_strong = any(kw in title for kw in STRONG_TITLE_KWS)
    current_is_soft = any(kw in title for kw in SOFT_TITLE_KWS)
    past_was_strong = any(any(kw in t for kw in STRONG_TITLE_KWS) for t in career_titles)
    
    if current_is_strong:
        title_score = 30.0
    elif current_is_soft and past_was_strong:
        title_score = 22.0
    elif current_is_soft:
        relevant_work_keywords = ['embedding', 'retrieval', 'ranking', 'vector', 'nlp',
                                   'recommendation', 'search', 'machine learning model',
                                   'deployed', 'production ml']
        career_evidence = sum(1 for kw in relevant_work_keywords if kw in career_descriptions)
        title_score = min(career_evidence * 3.0, 18.0)
    else:
        title_score = 0.0
        
    if title_score == 0.0:
        # Soft floor — candidate passed Phase 1, so give them a minimum base
        # and let the other components decide their rank. Matches rank.py logic.
        title_score = 5.0
    score += title_score
    
    # 2. Skills Quality
    skill_score = 0.0
    for skill in full_json.get('skills', []):
        name = skill.get('name', '').lower()
        relevance = SKILL_RELEVANCE.get(name, 0.0)
        if relevance == 0.0: continue
        
        prof_mult = PROFICIENCY_MULT.get(skill.get('proficiency', 'beginner'), 0.3)
        duration = skill.get('duration_months', 0)
        endorsements = skill.get('endorsements', 0)
        
        duration_trust = min(duration / 24.0, 1.0) if duration > 0 else 0.1
        endorsement_trust = 0.5 if endorsements == 0 else min(0.5 + 0.5 * math.log1p(endorsements) / math.log1p(50), 1.0)
        
        if skill.get('proficiency') == 'expert' and duration == 0:
            prof_mult = 0.1
            
        assessment_bonus = 0.0
        assessment_scores = full_json.get('redrob_signals', {}).get('skill_assessment_scores', {})
        for k, v in assessment_scores.items():
            if k.lower() == name:
                assessment_bonus = (v / 100.0) * 0.3
                
        skill_score += relevance * prof_mult * duration_trust * endorsement_trust * (1 + assessment_bonus)
    
    score += min(skill_score, 25.0)
    
    # 3. Experience Years
    yoe = full_json.get('profile', {}).get('years_of_experience', 0)
    if 5 <= yoe <= 9: exp_score = 15.0
    elif 4 <= yoe < 5: exp_score = 12.0
    elif 9 < yoe <= 11: exp_score = 10.0
    elif 3 <= yoe < 4: exp_score = 6.0
    elif 11 < yoe <= 13: exp_score = 5.0
    else: exp_score = 2.0
    score += exp_score
    
    # 4. Company Type
    career = full_json.get('career_history', [])
    total_months = sum(ch.get('duration_months', 0) for ch in career)
    consulting_months = sum(
        ch.get('duration_months', 0) for ch in career
        if any(firm in ch.get('company', '').lower() for firm in CONSULTING_FIRMS)
    )
    consulting_fraction = consulting_months / total_months if total_months > 0 else 0
    
    if consulting_fraction == 1.0: company_score = 0.0
    elif consulting_fraction > 0.7: company_score = 3.0
    elif consulting_fraction > 0.4: company_score = 7.0
    else: company_score = 15.0
    
    RELEVANT_INDUSTRIES = ['technology', 'software', 'e-commerce', 'fintech', 
                            'ai', 'machine learning', 'saas', 'internet', 
                            'food delivery', 'edtech', 'healthtech']
    has_relevant_product_co = any(
        any(ind in ch.get('industry', '').lower() for ind in RELEVANT_INDUSTRIES)
        and not any(firm in ch.get('company', '').lower() for firm in CONSULTING_FIRMS)
        for ch in career
    )
    if has_relevant_product_co:
        company_score = min(company_score + 3.0, 15.0)
    score += company_score
    
    # 5. Location
    country = full_json.get('profile', {}).get('country', '').lower()
    location = full_json.get('profile', {}).get('location', '').lower()
    willing_to_relocate = full_json.get('redrob_signals', {}).get('willing_to_relocate', False)
    TIER1_INDIA_CITIES = ['noida', 'pune', 'bengaluru', 'bangalore', 'hyderabad', 
                           'mumbai', 'delhi', 'gurgaon', 'gurugram', 'chennai', 'kolkata']
    
    if country == 'india':
        if any(city in location for city in ['noida', 'pune']): location_score = 10.0
        elif any(city in location for city in TIER1_INDIA_CITIES): location_score = 8.0
        else: location_score = 6.0 if willing_to_relocate else 5.0
    elif willing_to_relocate: location_score = 3.0
    else: location_score = 0.0
    score += location_score
    
    # 6. Behavioral Multiplier
    signals = full_json.get('redrob_signals', {})
    last_active_str = signals.get('last_active_date', '2020-01-01')
    try:
        last_active = date.fromisoformat(last_active_str)
    except:
        last_active = date(2020, 1, 1)
    days_inactive = (REFERENCE_DATE - last_active).days
    
    multiplier = 1.0
    if not signals.get('open_to_work_flag', False): multiplier *= 0.75
    
    if days_inactive <= 30: multiplier *= 1.0
    elif days_inactive <= 60: multiplier *= 0.95
    elif days_inactive <= 90: multiplier *= 0.85
    elif days_inactive <= 150: multiplier *= 0.70
    else: multiplier *= 0.50
    
    rrr = signals.get('recruiter_response_rate', 0.5)
    if rrr < 0.15: multiplier *= 0.60
    elif rrr < 0.30: multiplier *= 0.80
    elif rrr >= 0.60: multiplier *= 1.05
    
    icr = signals.get('interview_completion_rate', 0.7)
    if icr < 0.30: multiplier *= 0.75
    elif icr >= 0.80: multiplier *= 1.02
    
    notice = signals.get('notice_period_days', 60)
    if notice <= 30: multiplier *= 1.05
    elif notice <= 60: multiplier *= 1.0
    elif notice <= 90: multiplier *= 0.95
    else: multiplier *= 0.85
    
    gh = signals.get('github_activity_score', -1)
    if gh > 50: multiplier *= 1.05
    elif gh > 20: multiplier *= 1.02
    
    multiplier = max(0.40, min(multiplier, 1.15))
    final_score = score * multiplier

    # Scale from max theoretical score down to exactly 100.0
    return (final_score / MAX_RAW_SCORE) * 100.0

import rank

def run_ranker(progress=gr.Progress()):
    data_dir = "."
    jsonl_path = os.path.join(data_dir, "candidates.jsonl")
    
    if not os.path.exists(jsonl_path):
        return "⚠️ Could not find the 100k dataset at " + jsonl_path, ""

    progress(0.1, desc="Phase 1: Filtering 100,000 candidates...")
    # Phase 1: Fast Filter (Top 2000)
    top_2000 = rank.phase_1_filter(data_dir)
    top_ids = set(top_2000['candidate_id'].tolist())

    progress(0.4, desc="Phase 2: Deep scoring top 2,000 JSON records...")
    # Phase 2: Stream JSONL
    full_records = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            c = json.loads(line)
            if c.get('candidate_id') in top_ids:
                full_records[c['candidate_id']] = c
            if len(full_records) == len(top_ids):
                break
                
    progress(0.8, desc="Calculating final rankings...")
    scored_candidates = []
    for cid, record in full_records.items():
        score = score_candidate_full(record)
        if score > 0:
            p = record.get("profile", {})
            s = record.get("redrob_signals", {})
            skills_str = ", ".join(sk.get("name", "") for sk in record.get("skills", []) if sk.get("name"))
            
            scored_candidates.append({
                "id": cid,
                "title": p.get("current_title", ""),
                "experience": p.get("years_of_experience", 0),
                "skills": skills_str,
                "open_to_work": s.get("open_to_work_flag", True),
                "response_rate": s.get("recruiter_response_rate", 0),
                "notice_days": s.get("notice_period_days", 90),
                "score": score
            })

    ranked = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)[:10]

    rows = []
    for i, c in enumerate(ranked, 1):
        skills_short = ", ".join(c["skills"].split(", ")[:4])
        rows.append(
            f"**#{i} — {c['title']}** (`{c['id']}`) | {c['experience']}y exp | Score: `{c['score']:.3f}`\n"
            f"> Skills: {skills_short}\n"
            f"> Notice: {c['notice_days']}d | Response: {int(c['response_rate']*100)}% "
            f"| Open to work: {'✅' if c['open_to_work'] else '❌'}\n"
        )

    result = "\n---\n".join(rows)
    stats = (
        f"Ranked **100,000** actual candidates using the 2-Phase Rule-Based Pipeline!\n"
        f"Top candidate score: `{ranked[0]['score']:.3f}` | "
        f"#10 score: `{ranked[9]['score']:.3f}`"
    )
    return stats, result

# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="Redrob Candidate Ranker — Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 Redrob Candidate Ranker — Sandbox Demo")
    gr.Markdown(
        "This demo runs the 100% Rule-Based offline ranking pipeline on the **full 100,000 candidate dataset** "
        "from the challenge bundle. It streams and scores the entire dataset live in under 10 seconds.\n\n"
        "**No API calls are made during ranking** — pure algorithmic scoring."
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
        "---\n**Scoring formula:** 6 Components (Title Match, Skills Quality, Experience, Company Type, Location) * Behavioral Multiplier."
    )

if __name__ == "__main__":
    demo.launch()
