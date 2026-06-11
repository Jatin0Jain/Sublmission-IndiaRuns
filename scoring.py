import pandas as pd
import numpy as np

def score_career_trajectory(df: pd.DataFrame, parsed_jd: dict) -> pd.Series:
    """
    Calculates a career trajectory score (0-100) dynamically based on the JD's seniority requirement.
    Avoids hardcoded 'higher is better' logic by penalizing severe over/under qualification.
    """
    target_seniority = str(parsed_jd.get('seniority_level', 'mid')).lower()
    
    title_lower = df['current_title'].astype(str).str.lower()
    
    senior_keywords = ['senior', 'lead', 'principal', 'staff', 'manager', 'director', 'head', 'vp', 'chief']
    junior_keywords = ['junior', 'entry', 'intern', 'trainee', 'graduate']
    
    is_senior = title_lower.str.contains('|'.join(senior_keywords))
    is_junior = title_lower.str.contains('|'.join(junior_keywords))
    is_mid = ~is_senior & ~is_junior
    
    # Base score array
    scores = pd.Series(50, index=df.index, dtype=float)
    
    # Dynamic Seniority Alignment
    if target_seniority in ['senior', 'lead', 'executive']:
        scores[is_senior] = 100
        scores[is_mid] = 60
        scores[is_junior] = 20
    elif target_seniority in ['junior', 'entry']:
        scores[is_junior] = 100
        scores[is_mid] = 70
        scores[is_senior] = 40 # Overqualified penalty
    else: # mid level is default
        scores[is_mid] = 100
        scores[is_senior] = 80 # Slight overqualified penalty
        scores[is_junior] = 50 # Needs training
        
    # Experience adjustment (Bonus for hitting the "sweet spot" of the seniority level)
    exp = pd.to_numeric(df['experience_years'], errors='coerce').fillna(0)
    
    if target_seniority in ['senior', 'lead', 'executive']:
        exp_modifier = np.clip((exp - 5) * 5, -20, 20) # Bonus for > 5 yrs, penalty for < 5
    elif target_seniority in ['junior', 'entry']:
        exp_modifier = np.clip((2 - exp) * 5, -20, 10) # Bonus for < 2 yrs
    else:
        exp_modifier = np.clip(20 - abs(exp - 4) * 5, -20, 20) # Sweet spot around 4 years
        
    scores += exp_modifier
    
    return scores.clip(0, 100)

def score_activity(df: pd.DataFrame) -> pd.Series:
    """
    Calculates an activity score (0-100) using a smooth exponential decay curve 
    instead of rigid buckets. Prevents sudden score drops across bucket boundaries.
    """
    last_active = pd.to_datetime(df['last_active'], errors='coerce')
    now = pd.Timestamp.now()
    days_since = (now - last_active).dt.days
    days_since = days_since.fillna(365) # Default to 1 year if missing
    
    # Exponential decay: score drops smoothly over time
    # e^(-days / 130) -> 0 days = 100, 90 days ~ 50, 365 days ~ 6
    # This keeps scores highly organic and relatable
    scores = 100 * np.exp(-days_since / 130)
    
    # Ensure minimum baseline score of 10
    scores = np.maximum(scores, 10)
    
    return pd.Series(scores, index=df.index)

def compute_composite_score(df: pd.DataFrame, w_semantic: float, w_career: float, w_activity: float) -> pd.DataFrame:
    """
    Computes the final weighted composite score using vectorized Pandas math for max memory efficiency.
    No for-loops are used, processing 100k rows in milliseconds.
    """
    total_weight = w_semantic + w_career + w_activity
    if total_weight == 0:
        w_semantic, w_career, w_activity = 0.33, 0.33, 0.33
    else:
        w_semantic /= total_weight
        w_career /= total_weight
        w_activity /= total_weight

    sem_score = df.get('hybrid_score', df.get('semantic_score', pd.Series(0, index=df.index)))
    car_score = df.get('career_score', pd.Series(0, index=df.index))
    act_score = df.get('activity_score', pd.Series(0, index=df.index))
    
    df['composite_score'] = (
        (w_semantic * sem_score) + 
        (w_career * car_score) + 
        (w_activity * act_score)
    )
    
    df = df.sort_values(by='composite_score', ascending=False).reset_index(drop=True)
    return df

def apply_multi_signal_scoring(df: pd.DataFrame, parsed_jd: dict, weights: dict) -> pd.DataFrame:
    """
    Main orchestration function called by app.py. 
    Applies the individual scoring models and computes the weighted composite.
    """
    df = df.copy()
    
    # 1. Base semantic score (hybrid_score is 0-100, we scale it to 0-1)
    if 'hybrid_score' in df.columns:
        df['semantic_score'] = df['hybrid_score'] / 100.0
    else:
        df['semantic_score'] = 0.5
        
    # 2. Career trajectory (returns 0-100, we scale to 0-1 for radar)
    df['trajectory_score'] = score_career_trajectory(df, parsed_jd) / 100.0
    
    # 3. Activity score (returns 0-100, we scale to 0-1 for radar)
    df['activity_score'] = score_activity(df) / 100.0
    
    # 4. Composite score using the provided slider weights
    w_semantic = weights.get('semantic', 0.60)
    w_career = weights.get('career', 0.30)
    w_activity = weights.get('activity', 0.10)
    
    total_w = w_semantic + w_career + w_activity
    if total_w > 0:
        w_semantic /= total_w
        w_career /= total_w
        w_activity /= total_w
    
    # Calculate weighted sum (out of 100)
    df['composite_score'] = (
        (w_semantic * df['semantic_score'] * 100) + 
        (w_career * df['trajectory_score'] * 100) + 
        (w_activity * df['activity_score'] * 100)
    )
    
    return df.sort_values('composite_score', ascending=False).reset_index(drop=True)
