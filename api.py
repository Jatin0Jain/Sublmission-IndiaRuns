import os
from dotenv import load_dotenv
load_dotenv()
import time
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import shutil
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import search_core
import scoring
import llm_features

app = FastAPI(title="NovaSearch AI Backend")

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load FAISS Index
# DATA_DIR can be overridden via the DATA_DIR environment variable.
# Falls back to the default competition bundle layout if not set.
_default_data_dir = os.path.join(
    os.path.dirname(__file__),
    "[PUB] India_runs_data_and_ai_challenge",
    "India_runs_data_and_ai_challenge",
)
DATA_DIR = os.environ.get("DATA_DIR", _default_data_dir)
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "candidate_embeddings.npy")
CANDIDATES_PATH = os.path.join(DATA_DIR, "candidates_clean.parquet")

search_index = None
if os.path.exists(EMBEDDINGS_PATH):
    search_index = search_core.CandidateSearchIndex(CANDIDATES_PATH, EMBEDDINGS_PATH)

# --- Pydantic Models ---
class SearchRequest(BaseModel):
    jd_text: str
    skill_weight: float = 0.6
    growth_weight: float = 0.3
    activity_weight: float = 0.2
    demo_mode: bool = False

class CompareRequest(BaseModel):
    jd_text: str
    candidate_a: dict
    candidate_b: dict

class InterviewRequest(BaseModel):
    jd_text: str
    candidate: dict

def get_demo_data():
    return [
        {
            "name": "Sarah Jenkins", "title": "Senior Backend Engineer · Stripe", "score": 94, "years_exp": 6,
            "skills_matched": ["Python", "FastAPI", "Postgres", "Docker", "AWS"], "skills_missing": ["Kubernetes"],
            "narrative": "Sarah perfectly aligns with this role. She has 6 years of backend experience primarily using Python and Postgres, and recently led a major migration to AWS using Docker containers. Her progression from IC to tech lead in 2 years demonstrates exceptional growth velocity.",
            "caution": "No Kubernetes experience — may need ramp-up time for container orchestration.",
            "standout": "Promoted to Lead Engineer after just 2 years. Open-source contributor.",
            "radar": {"skill": 94, "experience": 88, "growth": 91, "culture": 76, "availability": 80},
            "interview_questions": [
                "Walk me through how you'd optimize a slow Postgres query on a 50M-row table with multiple JOINs.",
                "Design a microservices architecture for a real-time payment processing system.",
                "Describe your approach to containerizing a monolith — what breaks first?",
            ],
            "rank": 1
        },
        {
            "name": "David Chen", "title": "Backend Developer · Shopify", "score": 82, "years_exp": 5,
            "skills_matched": ["Python", "Postgres", "Docker"], "skills_missing": ["FastAPI", "AWS"],
            "narrative": "David is a strong Python developer with solid database skills. However, he lacks direct experience with FastAPI and cloud deployment on AWS, which are key requirements for this role.",
            "caution": "Missing explicit AWS and FastAPI experience; would require ramp-up time.",
            "standout": "Highly active contributor; immediately available. Strong open-source portfolio.",
            "radar": {"skill": 85, "experience": 70, "growth": 65, "culture": 80, "availability": 95},
            "interview_questions": [
                "Since you haven't used AWS directly, how would you approach learning our cloud stack?",
                "Compare Flask, Django REST, and FastAPI — when would you choose each?",
                "Describe a time you had to debug a production database issue under pressure.",
            ],
            "rank": 2
        },
        {
            "name": "Priya Sharma", "title": "Software Engineer III · Google", "score": 78, "years_exp": 4,
            "skills_matched": ["Python", "Docker", "AWS"], "skills_missing": ["FastAPI", "Postgres"],
            "narrative": "Priya brings strong cloud-native experience from Google with extensive AWS and Docker usage. Her gap is on the database side — she has primarily used BigQuery and Spanner rather than Postgres.",
            "caution": "No Postgres experience; primarily used proprietary Google databases.",
            "standout": "Built a production ML pipeline serving 1M+ daily requests.",
            "radar": {"skill": 78, "experience": 65, "growth": 88, "culture": 72, "availability": 70},
            "interview_questions": [
                "How does Postgres differ from BigQuery for OLTP workloads?",
                "Design a data migration strategy from Spanner to Postgres.",
                "Walk me through building a CI/CD pipeline for a Python microservice.",
            ],
            "rank": 3
        },
    ]


@app.post("/api/search")
async def run_search(req: SearchRequest):
    t0 = time.time()
    
    if req.demo_mode:
        time.sleep(1.0)
        return {
            "candidates": get_demo_data(),
            "times": {"vector": 0.48, "ai": 0.72}
        }
        
    if search_index is None:
        raise HTTPException(status_code=500, detail="Search engine not loaded (embeddings missing).")
        
    try:
        jd_reqs = search_core.extract_jd_requirements(req.jd_text)
        top_k = search_index.semantic_search(req.jd_text, top_k=50)
        df_50 = pd.DataFrame(top_k)
        
        t_vector = time.time() - t0
        t1 = time.time()
        
        w = {
            "semantic": req.skill_weight, 
            "career": req.growth_weight, 
            "activity": req.activity_weight
        }
        
        df_scored = scoring.apply_multi_signal_scoring(df_50, jd_reqs, weights=w)
        df_top = df_scored.sort_values("composite_score", ascending=False).head(25)
        
        evals = llm_features.evaluate_and_rerank(df_top, req.jd_text)
        
        candidates = []
        for ev in evals:
            try:
                row = df_top[df_top["candidate_id"] == ev["candidate_id"]].iloc[0]
            except IndexError:
                continue
            
            skills_str = str(row["skills"]).lower()
            matched = [s for s in jd_reqs.get("required_skills", []) if s.lower() in skills_str]
            missed  = [s for s in jd_reqs.get("required_skills", []) if s.lower() not in skills_str]
            
            candidates.append({
                "candidate_id": ev["candidate_id"],
                "name": row["name"], 
                "title": row["current_title"],
                "score": int(row["composite_score"]), 
                "years_exp": row["experience_years"],
                "skills_matched": matched, 
                "skills_missing": missed,
                "narrative": ev["match_narrative"],
                "caution": ev["red_flag"], 
                "standout": ev["green_flag"],
                "radar": {
                    "skill": int(row["semantic_score"] * 100),
                    "experience": min(100, int(row["experience_years"] * 10)),
                    "growth": int(row["trajectory_score"] * 100),
                    "culture": 80,
                    "availability": int(row["activity_score"] * 100),
                },
                "interview_questions": [],
                "rank": 0
            })
            
        # Sort by actual score instead of LLM's rank for UI consistency
        candidates.sort(key=lambda x: x["score"], reverse=True)
        for i, c in enumerate(candidates, 1):
            c["rank"] = i
            
        t_ai = time.time() - t1
        
        return {
            "candidates": candidates,
            "times": {"vector": t_vector, "ai": t_ai}
        }
        
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            raise HTTPException(status_code=429, detail="Gemini API rate limit reached. Please wait or use Demo Mode.")
        else:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/compare")
async def run_compare(req: CompareRequest):
    try:
        comparison = llm_features.compare_candidates(req.candidate_a, req.candidate_b, req.jd_text)
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/interview_prep")
async def run_interview_prep(req: InterviewRequest):
    try:
        questions = llm_features.generate_interview_questions(req.candidate, req.jd_text)
        return {"questions": questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload_dataset")
async def upload_dataset(file: UploadFile = File(...), mode: str = Form("merge")):
    global search_index
    try:
        # Save temp file
        ext = os.path.splitext(file.filename)[1].lower()
        tmp_path = f"temp_dataset{ext}"
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Load dataset
        if ext == ".csv":
            df = pd.read_csv(tmp_path)
        elif ext == ".parquet":
            df = pd.read_parquet(tmp_path)
        elif ext in [".json", ".jsonl"]:
            df = pd.read_json(tmp_path, lines=(ext==".jsonl"))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use CSV, JSON, or Parquet.")
            
        # Basic validation
        if 'name' not in df.columns or 'skills' not in df.columns:
            raise HTTPException(status_code=400, detail="Dataset must contain 'name' and 'skills' columns")
            
        # Fill missing columns for compatibility
        if 'experience_years' not in df.columns: df['experience_years'] = 5
        if 'current_title' not in df.columns: df['current_title'] = 'Unknown Title'
        if 'bio' not in df.columns: df['bio'] = ''
        if 'last_active' not in df.columns: df['last_active'] = pd.Timestamp.now().strftime('%Y-%m-%d')
            
        # Build profile text
        df['profile_text'] = df.apply(lambda row: f"Title: {row.get('current_title', '')}. Skills: {row.get('skills', '')}. Experience: {row.get('experience_years', 0)} years. Bio: {row.get('bio', '')}", axis=1)
        
        # Save the cleaned dataset for the CandidateSearchIndex to read
        clean_path = "temp_dataset_clean.parquet"
        df.to_parquet(clean_path)
        
        # Generate embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(df['profile_text'].tolist(), show_progress_bar=False, batch_size=256)
        
        # Merge or Replace
        if mode == "replace" or search_index is None:
            tmp_emb_path = "temp_embeddings.npy"
            np.save(tmp_emb_path, embeddings)
            search_index = search_core.CandidateSearchIndex(clean_path, tmp_emb_path)
            return {"message": f"Successfully replaced index. Encoded {len(df)} candidates."}
        else:
            search_index.add_candidates(df, embeddings)
            return {"message": f"Successfully merged {len(df)} candidates into existing pool."}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset_dataset")
async def reset_dataset():
    global search_index
    try:
        if os.path.exists(EMBEDDINGS_PATH):
            search_index = search_core.CandidateSearchIndex(CANDIDATES_PATH, EMBEDDINGS_PATH)
            return {"message": "Successfully restored the default 100k candidate dataset."}
        else:
            raise HTTPException(status_code=404, detail="Default embeddings not found on disk.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
