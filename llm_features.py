import google.generativeai as genai
import json
import os
import time as _time
import pandas as pd
from pydantic import BaseModel, Field

def get_model():
    """Initializes and returns the Gemini 1.5 Flash model."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing! Please set it before running.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')

# -----------------------------------------------------------------------------
# 1. Holistic Re-ranker, Narratives, and Red/Green Flags (ALL IN ONE CALL)
# -----------------------------------------------------------------------------
class CandidateEvaluation(BaseModel):
    candidate_id: str
    rank: int = Field(description="Rank from 1 to 10")
    match_narrative: str = Field(description="2-3 sentence recruiter-quality explanation of why they are a strong fit.")
    green_flag: str = Field(description="One standout positive signal.")
    red_flag: str = Field(description="One cautionary signal or explicitly missing skill. Do not invent skills.")

class RerankResponse(BaseModel):
    evaluations: list[CandidateEvaluation] = Field(description="Top 10 candidates only")

def evaluate_and_rerank(top_25_df: pd.DataFrame, jd_text: str) -> list[dict]:
    """
    Takes the top 25 candidates, passes them to Gemini, and returns the top 10 
    re-ranked with narratives and flags. Using Structured Outputs saves us from
    making 10 separate API calls for narratives!
    """
    model = get_model()
    
    # Format candidates compactly to save tokens
    candidates_text = ""
    for _, row in top_25_df.iterrows():
        candidates_text += f"ID: {row['candidate_id']} | Name: {row['name']} | Title: {row['current_title']} | Exp: {row['experience_years']}y\n"
        candidates_text += f"Skills: {row['skills']}\nBio: {row['bio']}\n---\n"
        
    prompt = f"""
    You are an expert technical recruiter. You have a Job Description and a shortlist of 25 candidates.
    Your task is to select the TOP 10 best candidates, rank them 1 to 10, and provide an evaluation for each.
    
    Job Description:
    {jd_text}
    
    Candidates:
    {candidates_text}
    
    CRITICAL: 
    - Only evaluate your top 10 choices.
    - Do not hallucinate skills. Base your evaluation strictly on the provided text.
    - Be brutal with the 'red_flag' if they lack a core requirement.
    """
    
    for attempt in range(3):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=RerankResponse,
                    temperature=0.1
                )
            )
            return json.loads(response.text).get("evaluations", [])
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                _time.sleep(10 * (attempt + 1))
                continue
            raise

# -----------------------------------------------------------------------------
# 2. Side-by-Side Candidate Comparison
# -----------------------------------------------------------------------------
class CandidateComparison(BaseModel):
    technical_winner: str = Field(description="Candidate A, Candidate B, or Tie")
    technical_reasoning: str
    experience_winner: str = Field(description="Candidate A, Candidate B, or Tie")
    experience_reasoning: str
    overall_recommendation: str = Field(description="Brief final verdict on who to hire")

def compare_candidates(candidate_a: dict, candidate_b: dict, jd_text: str) -> dict:
    """Side-by-side comparison of two specific candidates."""
    model = get_model()
    
    prompt = f"""
    Compare Candidate A ({candidate_a.get('name')}) and Candidate B ({candidate_b.get('name')}) 
    for the following Job Description. Be highly objective.
    
    Job Description:
    {jd_text}
    
    Candidate A: {json.dumps(candidate_a)}
    Candidate B: {json.dumps(candidate_b)}
    """
    
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=CandidateComparison,
            temperature=0.2
        )
    )
    return json.loads(response.text)

# -----------------------------------------------------------------------------
# 3. Interview Prep Generator
# -----------------------------------------------------------------------------
class InterviewPrep(BaseModel):
    questions: list[str] = Field(description="Exactly 3 highly targeted technical questions")

def generate_interview_questions(candidate: dict, jd_text: str) -> list[str]:
    """Generates 3 interview questions specifically targeting the candidate's weak spots."""
    model = get_model()
    
    prompt = f"""
    You are preparing a hiring manager to interview this candidate for this Job Description.
    Identify the weakest areas or missing skills for this candidate based on the JD.
    Generate exactly 3 highly technical interview questions to test those specific weak spots.
    
    Job Description:
    {jd_text}
    
    Candidate Profile: {json.dumps(candidate)}
    """
    
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=InterviewPrep,
            temperature=0.3
        )
    )
    
    return json.loads(response.text).get("questions", [])
