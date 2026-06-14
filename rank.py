import argparse
import pandas as pd
import gzip
import json
import os
import csv
import math
from datetime import date

REFERENCE_DATE = date(2026, 6, 14)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()

# =======================================================
# PHASE 1: FAST FILTER
# =======================================================
GOOD_TITLE_KEYWORDS = [
    'ml engineer', 'machine learning engineer', 'ai engineer', 'nlp engineer',
    'data scientist', 'applied scientist', 'research engineer', 'search engineer',
    'recommendation', 'retrieval', 'ranking engineer', 'applied ml',
    'deep learning engineer', 'ai researcher', 'mlops engineer'
]
OK_TITLE_KEYWORDS = [
    'software engineer', 'backend engineer', 'data engineer',
    'platform engineer', 'senior engineer', 'tech lead'
]
CONSULTING_FIRMS = [
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini', 
    'hcl', 'tech mahindra', 'mindtree', 'hexaware', 'mphasis', 'ltimindtree'
]
AI_SKILLS = {
    'tier_a': ['faiss', 'pinecone', 'qdrant', 'milvus', 'weaviate', 'opensearch',
               'elasticsearch', 'sentence transformers', 'sentence-transformers',
               'embeddings', 'vector search', 'hybrid search', 'bge', 'e5 embeddings'],
    'tier_b': ['nlp', 'information retrieval', 'learning to rank', 'ltr', 'xgboost',
               'transformers', 'hugging face', 'bert', 'rag', 'retrieval augmented',
               'fine-tuning', 'lora', 'qlora', 'peft', 'ranking', 'recommendation systems'],
    'tier_c': ['python', 'pytorch', 'tensorflow', 'scikit-learn', 'sklearn',
               'machine learning', 'deep learning', 'llm', 'gpt', 'mlops', 'mlflow']
}

def title_category(title):
    t = str(title).lower()
    if any(kw in t for kw in GOOD_TITLE_KEYWORDS):
        return 'strong'
    if any(kw in t for kw in OK_TITLE_KEYWORDS):
        return 'possible'
    return 'disqualified'

def experience_score_coarse(yoe):
    if 5 <= yoe <= 9: return 1.0
    elif 4 <= yoe < 5 or 9 < yoe <= 11: return 0.7
    elif 3 <= yoe < 4 or 11 < yoe <= 13: return 0.4
    else: return 0.1

def coarse_skill_score(skills_str):
    s = str(skills_str).lower()
    score = sum(2.0 for kw in AI_SKILLS['tier_a'] if kw in s)
    score += sum(1.5 for kw in AI_SKILLS['tier_b'] if kw in s)
    score += sum(0.5 for kw in AI_SKILLS['tier_c'] if kw in s)
    return min(score, 20.0)

def phase_1_filter(data_dir):
    parquet_path = os.path.join(data_dir, "candidates_clean.parquet")
    if not os.path.exists(parquet_path):
        parquet_path = "candidates_clean.parquet"
    
    df = pd.read_parquet(parquet_path)
    df['title_cat'] = df['current_title'].apply(title_category)
    df['exp_score'] = df['experience_years'].apply(experience_score_coarse)
    df['coarse_skill_score'] = df['skills'].apply(coarse_skill_score)

    candidates_to_score = df[
        (df['title_cat'] == 'strong') |
        ((df['title_cat'] == 'possible') & (df['coarse_skill_score'] >= 5))
    ].copy()

    candidates_to_score['coarse_total'] = (
        candidates_to_score['coarse_skill_score'] * 0.5 +
        candidates_to_score['exp_score'] * 5
    )
    return candidates_to_score.nlargest(2000, 'coarse_total')

# =======================================================
# PHASE 2: DEEP SCORER
# =======================================================
SKILL_RELEVANCE = {
    'faiss': 3.0, 'pinecone': 3.0, 'qdrant': 3.0, 'milvus': 3.0,
    'weaviate': 3.0, 'opensearch': 2.5, 'elasticsearch': 2.5,
    'sentence transformers': 3.0, 'sentence-transformers': 3.0,
    'embeddings': 2.5, 'vector search': 3.0, 'hybrid search': 3.0,
    'information retrieval': 3.0, 'bge': 2.5, 'e5': 2.0,
    'nlp': 2.0, 'learning to rank': 2.5, 'ltr': 2.5,
    'transformers': 2.0, 'hugging face transformers': 2.0,
    'bert': 1.5, 'rag': 2.0, 'fine-tuning llms': 2.0, 'fine-tuning': 1.5,
    'lora': 2.0, 'qlora': 2.0, 'peft': 2.0,
    'xgboost': 1.5, 'ranking': 2.0, 'recommendation systems': 2.5,
    'mlflow': 1.0, 'mlops': 1.5, 'feature engineering': 1.5,
    'python': 1.5, 'pytorch': 1.0, 'tensorflow': 0.8,
    'scikit-learn': 0.8, 'machine learning': 0.8, 'deep learning': 0.8,
    'llm': 1.0, 'gpt': 0.5, 'llama': 0.8,
    'spark': 0.5, 'kafka': 0.3, 'aws': 0.3, 'gcp': 0.3
}
PROFICIENCY_MULT = {'beginner': 0.3, 'intermediate': 0.6, 'advanced': 0.85, 'expert': 1.0}

def score_candidate_full(cid, full_json):
    score = 0.0
    
    # 1. Title + Career Match
    title = full_json['profile'].get('current_title', '').lower()
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
        return 0.0
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
    yoe = full_json['profile'].get('years_of_experience', 0)
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
    country = full_json['profile'].get('country', '').lower()
    location = full_json['profile'].get('location', '').lower()
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
    return score * multiplier

def generate_reasoning(cid, full_json, rank, score, pre_generated):
    if cid in pre_generated:
        return pre_generated[cid]
    
    p = full_json.get('profile', {})
    sig = full_json.get('redrob_signals', {})
    
    title = p.get('current_title', 'Engineer')
    yoe = p.get('years_of_experience', 0)
    company = p.get('current_company', 'Tech Co')
    country = p.get('country', 'Unknown')
    
    relevant_skill_names = [s.get('name', '') for s in full_json.get('skills', []) 
                            if s.get('name', '').lower() in SKILL_RELEVANCE and s.get('duration_months', 0) > 6]
    top_skills = relevant_skill_names[:2] if relevant_skill_names else ['general ML']
    
    notice = sig.get('notice_period_days', 30)
    rrr = sig.get('recruiter_response_rate', 0.5)
    open_to = sig.get('open_to_work_flag', True)
    
    concerns = []
    if notice > 60: concerns.append(f"{notice}-day notice period")
    if country.lower() != 'india': concerns.append(f"based outside India ({country})")
    if rrr < 0.30: concerns.append(f"low recruiter response rate ({rrr:.0%})")
    if not open_to: concerns.append("not currently marked open to work")
    
    skills_str = ' and '.join(top_skills) if top_skills else 'adjacent ML skills'
    concern_str = f"; concern: {', '.join(concerns)}" if concerns else ""
    
    return f"{title} with {yoe:.1f} years at {company}; strong in {skills_str}{concern_str}."

def main():
    args = parse_args()
    data_dir = os.path.dirname(args.candidates)
    if not data_dir: data_dir = "."
    
    print("Phase 1: Fast filtering top 2000 from parquet...")
    top_2000 = phase_1_filter(data_dir)
    top_ids = set(top_2000['candidate_id'].tolist())
    
    print("Phase 2: Deep scoring from JSONL...")
    full_records = {}
    
    if args.candidates.endswith('.gz'):
        f = gzip.open(args.candidates, 'rt', encoding='utf-8')
    else:
        f = open(args.candidates, 'r', encoding='utf-8')
        
    for line in f:
        if not line.strip(): continue
        c = json.loads(line)
        if c.get('candidate_id') in top_ids:
            full_records[c['candidate_id']] = c
        if len(full_records) == len(top_ids):
            break
    f.close()
    
    scored_candidates = []
    for cid, record in full_records.items():
        score = score_candidate_full(cid, record)
        if score > 0:
            scored_candidates.append({
                'candidate_id': cid,
                'score': score,
                'full_json': record
            })
            
    results = sorted(scored_candidates, key=lambda x: x['score'], reverse=True)[:100]
    
    print("Loading reasonings...")
    pre_generated = {}
    reasonings_file = os.path.join(data_dir, "candidate_reasonings.json")
    if not os.path.exists(reasonings_file):
        reasonings_file = "candidate_reasonings.json"
    if os.path.exists(reasonings_file):
        with open(reasonings_file, 'r', encoding='utf-8') as rf:
            pre_generated = json.load(rf)
            
    rows = []
    for i, r in enumerate(results):
        rows.append({
            'candidate_id': r['candidate_id'],
            'rank': i + 1,
            'score': round(r['score'], 4),
            'reasoning': generate_reasoning(r['candidate_id'], r['full_json'], i + 1, r['score'], pre_generated)
        })
        
    print("Writing submission.csv...")
    with open(args.out, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=['candidate_id', 'rank', 'score', 'reasoning'])
        writer.writeheader()
        writer.writerows(rows)
        
    print("Done! Validating...")
    # Optional inline validation or just exit
    
if __name__ == "__main__":
    main()
