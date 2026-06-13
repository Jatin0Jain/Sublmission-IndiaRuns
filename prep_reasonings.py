import os
import json
import time
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel, Field

load_dotenv()

# Setup Gemini
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

class Reasoning(BaseModel):
    candidate_id: str
    reasoning: str = Field(description="1-2 sentence professional recruiter explanation of why this candidate is a good fit, explicitly calling out actual facts from their profile and noting any honest concerns like missing skills or notice period.")

class ReasoningBatch(BaseModel):
    results: list[Reasoning]

def calculate_behavioral_score(row):
    score = 0.0
    exp = row.get("experience_years", 0)
    if 4 <= exp <= 10: score += 0.3
    elif exp > 10: score += 0.2
    else: score += 0.1
        
    resp = row.get("recruiter_response_rate", 0)
    score += (resp * 0.3)
    
    try: days_inactive = (pd.Timestamp.now() - row["last_active"]).days
    except: days_inactive = 30
        
    if days_inactive < 7: score += 0.2
    elif days_inactive < 30: score += 0.1
        
    np_days = row.get("notice_period_days", 30)
    if np_days <= 30: score += 0.2
    elif np_days <= 60: score += 0.1
    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with candidates_clean.parquet and candidate_embeddings.npy")
    args = parser.parse_args()
    data_dir = args.data_dir

    print("Loading data to find top candidates...", flush=True)
    
    parquet_path = os.path.join(data_dir, "candidates_clean.parquet")
    embeddings_path = os.path.join(data_dir, "candidate_embeddings.npy")
    
    df = pd.read_parquet(parquet_path)
    embeddings = np.load(embeddings_path)
    
    jd_query = """Senior AI Engineer Founding Team Redrob AI. Deep technical depth in modern ML systems, embeddings, retrieval, ranking, LLMs, fine-tuning. Production experience with embeddings-based retrieval systems sentence-transformers, OpenAI embeddings, BGE. Production experience with vector databases Pinecone, Weaviate, Qdrant, Milvus, FAISS. Strong Python code quality. Hands-on experience designing evaluation frameworks NDCG, MRR, MAP, offline-to-online correlation, A/B testing. 5 to 9 years experience. Scrappy product-engineering attitude."""
    
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = st_model.encode(jd_query)
    
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    similarities = np.dot(embeddings_norm, query_norm)
    
    df["semantic_score"] = similarities
    df = df[df["is_honeypot"] == 0].copy()
    df["behavioral_score"] = df.apply(calculate_behavioral_score, axis=1)
    df["final_score"] = (df["semantic_score"] * 0.7) + (df["behavioral_score"] * 0.3)
    
    # Take top 100 for the final submission
    top_candidates = df.sort_values(by="final_score", ascending=False).head(100).copy()
    
    print("Generating LLM reasonings in batches of 20...", flush=True)
    reasonings_dict = {}
    batch_size = 20
    candidates_list = top_candidates.to_dict('records')
    
    for i in range(0, len(candidates_list), batch_size):
        batch = candidates_list[i:i+batch_size]
        
        batch_text = ""
        for c in batch:
            batch_text += f"ID: {c['candidate_id']} | Exp: {c['experience_years']}y | Title: {c['current_title']} | Skills: {c['skills']} | Response Rate: {int(c['recruiter_response_rate']*100)}% | Notice: {c['notice_period_days']}d\n"
            
        prompt = f"""
        You are an expert technical recruiter evaluating candidates for this Job Description:
        {jd_query}
        
        For each candidate below, write a 1-2 sentence reasoning explaining why they are a strong fit. 
        CRITICAL RULES:
        1. Reference specific facts (years of exp, specific skills, response rate).
        2. Do not use templates. Make them varied.
        3. If they lack something (e.g., notice period > 30 days, or slightly under experience), acknowledge it as an 'honest concern'.
        4. Do NOT hallucinate skills not listed.
        
        Candidates:
        {batch_text}
        """
        
        for attempt in range(3):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=ReasoningBatch,
                        temperature=0.4
                    )
                )
                res_data = json.loads(response.text)
                for item in res_data.get("results", []):
                    reasonings_dict[item["candidate_id"]] = item["reasoning"]
                print(f"Processed batch {i//batch_size + 1}", flush=True)
                
                # Incrementally save to data_dir so generate_submission.py can find it
                reasonings_path = os.path.join(data_dir, "candidate_reasonings.json")
                with open(reasonings_path, "w", encoding="utf-8") as f:
                    json.dump(reasonings_dict, f, indent=2)
                    
                time.sleep(2) # Avoid rate limits
                break
            except Exception as e:
                print(f"Error on attempt {attempt}: {e}", flush=True)
                time.sleep(10)
                
    print(f"Saved {len(reasonings_dict)} reasonings to {os.path.join(data_dir, 'candidate_reasonings.json')}", flush=True)

if __name__ == "__main__":
    main()
