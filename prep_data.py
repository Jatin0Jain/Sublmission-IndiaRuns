import pandas as pd
import json
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Flatten candidates.jsonl into a clean parquet file.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing candidates.jsonl")
    return parser.parse_args()

def detect_honeypot(record):
    """Multi-heuristic honeypot detector."""
    is_honeypot = 0
    profile = record.get("profile", {})
    years_exp = profile.get("years_of_experience", 0)
    signals = record.get("redrob_signals", {})
    skills = record.get("skills", [])

    # Check 1: Expert proficiency with 0 months usage (impossible)
    expert_zero_duration = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    )
    if expert_zero_duration > 3:
        is_honeypot = 1

    # Check 2: Total career duration wildly inconsistent with claimed experience
    career = record.get("career_history", [])
    total_career_months = sum(c.get("duration_months", 0) for c in career)
    if years_exp > 2 and total_career_months < (years_exp * 12) * 0.2:
        is_honeypot = 1

    # Check 3: Profile completeness nearly zero — trap profile
    if signals.get("profile_completeness_score", 100) < 10:
        is_honeypot = 1

    # Check 4: Interview completion is exactly 0 but has many applications (ghost)
    interview_rate = signals.get("interview_completion_rate", 1.0)
    apps = signals.get("applications_submitted_30d", 0)
    if interview_rate == 0.0 and apps > 5:
        is_honeypot = 1

    # Check 5: Implausible salary expectations (e.g. 0 salary)
    salary = signals.get("expected_salary_range_inr_lpa", {})
    if salary.get("max", 1) == 0:
        is_honeypot = 1

    return is_honeypot


def flatten_candidate(record):
    """Flattens a single nested JSON candidate record into a flat dictionary."""
    profile = record.get("profile", {})
    signals = record.get("redrob_signals", {})

    # Skills as comma-separated string
    skills_list = record.get("skills", [])
    skills_str = ", ".join([s.get("name", "") for s in skills_list if s.get("name")])

    # Career history text (role descriptions — key for embedding)
    career = record.get("career_history", [])
    career_text = " | ".join([
        f"{c.get('title', '')} at {c.get('company', '')}: {c.get('description', '')}"
        for c in career[:4]  # limit to last 4 roles to cap token length
    ])

    # Salary range
    salary = signals.get("expected_salary_range_inr_lpa", {})

    return {
        # Core profile
        "candidate_id": record.get("candidate_id"),
        "name": profile.get("anonymized_name", "Unknown"),
        "current_title": profile.get("current_title", "Unknown"),
        "company": profile.get("current_company", "Unknown"),
        "experience_years": profile.get("years_of_experience", 0),
        "bio": profile.get("summary", ""),
        "skills": skills_str,
        "career_text": career_text,

        # Redrob signals (all 23)
        "profile_completeness_score": signals.get("profile_completeness_score", 0),
        "last_active": signals.get("last_active_date", "1970-01-01"),
        "open_to_work_flag": signals.get("open_to_work_flag", False),
        "profile_views_30d": signals.get("profile_views_received_30d", 0),
        "applications_30d": signals.get("applications_submitted_30d", 0),
        "recruiter_response_rate": signals.get("recruiter_response_rate", 0.0),
        "avg_response_time_hours": signals.get("avg_response_time_hours", 999),
        "connection_count": signals.get("connection_count", 0),
        "endorsements_received": signals.get("endorsements_received", 0),
        "notice_period_days": signals.get("notice_period_days", 90),
        "salary_min_lpa": salary.get("min", 0),
        "salary_max_lpa": salary.get("max", 0),
        "preferred_work_mode": signals.get("preferred_work_mode", "flexible"),
        "willing_to_relocate": signals.get("willing_to_relocate", False),
        "github_activity_score": signals.get("github_activity_score", -1),
        "search_appearance_30d": signals.get("search_appearance_30d", 0),
        "saved_by_recruiters_30d": signals.get("saved_by_recruiters_30d", 0),
        "interview_completion_rate": signals.get("interview_completion_rate", 0.5),
        "offer_acceptance_rate": signals.get("offer_acceptance_rate", -1),
        "verified_email": signals.get("verified_email", False),
        "verified_phone": signals.get("verified_phone", False),
        "linkedin_connected": signals.get("linkedin_connected", False),

        # Honeypot flag
        "is_honeypot": detect_honeypot(record),
    }


def process_data(data_dir):
    input_file = os.path.join(data_dir, "candidates.jsonl")
    output_file = os.path.join(data_dir, "candidates_clean.parquet")

    print(f"Reading and flattening data from {input_file}...")
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(flatten_candidate(record))
            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1} records...")

    df = pd.DataFrame(records)

    print("Handling missing values and data types...")
    df['experience_years'] = df['experience_years'].fillna(0)
    df['bio'] = df['bio'].fillna('')
    df['skills'] = df['skills'].fillna('')
    df['career_text'] = df['career_text'].fillna('')
    df['last_active'] = pd.to_datetime(df['last_active'], errors='coerce')
    df['last_active'] = df['last_active'].fillna(pd.Timestamp.now() - pd.DateOffset(years=2))
    df['open_to_work_flag'] = df['open_to_work_flag'].fillna(False)
    df['willing_to_relocate'] = df['willing_to_relocate'].fillna(False)
    df['verified_email'] = df['verified_email'].fillna(False)
    df['verified_phone'] = df['verified_phone'].fillna(False)
    df['linkedin_connected'] = df['linkedin_connected'].fillna(False)

    honeypot_count = df['is_honeypot'].sum()
    print(f"Honeypots detected: {honeypot_count} ({honeypot_count/len(df)*100:.1f}%)")

    print(f"Saving cleaned dataset to {output_file}...")
    df.to_parquet(output_file, index=False)
    print(f"Data preparation complete! Dataset shape: {df.shape}")


if __name__ == "__main__":
    args = parse_args()
    process_data(args.data_dir)
