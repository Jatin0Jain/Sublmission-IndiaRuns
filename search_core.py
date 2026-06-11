import google.generativeai as genai
import json
import os
import time as _time
from pydantic import BaseModel, Field
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

# Cache the embedding model so it's only loaded once
_embed_model = None
def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embed_model

# Improvement: Use Pydantic schema to force Gemini into returning PERFECT JSON every time.
# This completely eliminates the "JSON parsing breaks" failure point mentioned in the plan.
class ParsedJD(BaseModel):
    required_skills: list[str] = Field(description="Must-have skills")
    nice_to_have_skills: list[str] = Field(description="Bonus skills")
    seniority_level: str = Field(description="junior, mid, senior, lead, or executive")
    culture_signals: list[str] = Field(description="Cultural traits mentioned")
    deal_breakers: list[str] = Field(description="Strict requirements or disqualifiers")

def parse_job_description(jd_text: str) -> dict:
    """Uses Gemini 1.5 Flash structured outputs to parse the JD."""
    # Assuming GEMINI_API_KEY is in environment variables
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing!")
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    You are an expert technical recruiter. Analyze the following job description and extract 
    the core requirements exactly matching the JSON schema requested.
    
    Job Description:
    {jd_text}
    """
    
    for attempt in range(3):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=ParsedJD,
                    temperature=0.1
                )
            )
            return json.loads(response.text)
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                _time.sleep(10 * (attempt + 1))
                continue
            raise

def build_faiss_index(embeddings: np.ndarray):
    """Builds a FAISS index for fast vector search."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # Inner Product (equivalent to Cosine similarity if vectors are normalized)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def perform_hybrid_search(
    jd_embedding: np.ndarray, 
    parsed_jd: dict, 
    df: pd.DataFrame, 
    faiss_index, 
    top_k: int = 50
):
    """
    Combines Vector Search (FAISS) with Keyword Search on required skills.
    This is the "Hybrid Search" improvement from the pitch notes.
    """
    # 1. Vector Search
    jd_embedding_normalized = np.copy(jd_embedding)
    faiss.normalize_L2(jd_embedding_normalized)
    
    # Search for more candidates initially so we can re-rank them with the hybrid score
    # Increased multiplier from 4 to 40 so keyword boost has a larger pool to find rare skills like 'Python'
    similarities, indices = faiss_index.search(jd_embedding_normalized, top_k * 40) 
    
    vector_results = df.iloc[indices[0]].copy()
    vector_results['semantic_score'] = similarities[0] * 100 # Convert to percentage
    
    # 2. Keyword boost (Simple exact matching on required skills)
    required_skills = [s.lower() for s in parsed_jd.get('required_skills', [])]
    
    def calculate_skill_match(skills_str):
        if not required_skills: return 0
        candidate_skills = str(skills_str).lower()
        matches = sum(1 for skill in required_skills if skill in candidate_skills)
        return (matches / len(required_skills)) * 100

    vector_results['keyword_score'] = vector_results['skills'].apply(calculate_skill_match)
    
    # 3. Combine scores (70% semantic, 30% keyword)
    vector_results['hybrid_score'] = (vector_results['semantic_score'] * 0.7) + (vector_results['keyword_score'] * 0.3)
    
    # Sort by the new hybrid score and take the final top_k
    final_results = vector_results.sort_values(by='hybrid_score', ascending=False).head(top_k)
    return final_results

# ALIAS for app.py
extract_jd_requirements = parse_job_description

class CandidateSearchIndex:
    def __init__(self, data_path: str, embeddings_path: str):
        # Support loading Parquet, JSONL, or JSON
        if data_path.endswith('.parquet'):
            self.df = pd.read_parquet(data_path)
        elif data_path.endswith('.jsonl'):
            self.df = pd.read_json(data_path, lines=True)
        else:
            self.df = pd.read_json(data_path)
            
        self.embeddings = np.load(embeddings_path)
        self.faiss_index = build_faiss_index(self.embeddings)
        
    def add_candidates(self, new_df: pd.DataFrame, new_embeddings: np.ndarray):
        """Merges new candidates into the existing index."""
        # Ensure dimensions match
        if new_embeddings.shape[1] != self.embeddings.shape[1]:
            raise ValueError("Embedding dimensions do not match.")
            
        # Append DataFrame
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        
        # Append Embeddings array
        self.embeddings = np.vstack((self.embeddings, new_embeddings))
        
        # Add to FAISS Index
        faiss.normalize_L2(new_embeddings)
        self.faiss_index.add(new_embeddings)
        
    def semantic_search(self, jd_text: str, top_k: int = 50) -> list[dict]:
        parsed_jd = parse_job_description(jd_text)
        
        # Use the SAME model that built the embeddings (all-MiniLM-L6-v2)
        model = _get_embed_model()
        jd_embedding = model.encode([jd_text])
        jd_embedding = np.array(jd_embedding, dtype='float32')
        
        # Perform hybrid search
        results_df = perform_hybrid_search(
            jd_embedding, 
            parsed_jd, 
            self.df, 
            self.faiss_index, 
            top_k=top_k
        )
        return results_df.to_dict(orient='records')
