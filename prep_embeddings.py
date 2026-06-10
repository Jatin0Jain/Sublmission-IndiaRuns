import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Generate candidate embeddings.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing candidates_clean.parquet")
    return parser.parse_args()


def generate_embeddings(data_dir):
    parquet_file = os.path.join(data_dir, "candidates_clean.parquet")
    embeddings_file = os.path.join(data_dir, "candidate_embeddings.npy")

    print(f"Loading {parquet_file}...")
    df = pd.read_parquet(parquet_file)

    print("Building rich text representation for embedding...")
    # Include career_text (role descriptions) which is the strongest signal for
    # "production experience" — the key JD requirement.
    df['profile_text'] = (
        "Title: " + df['current_title'].fillna('') + ". " +
        "Skills: " + df['skills'].fillna('') + ". " +
        "Experience: " + df['experience_years'].astype(str) + " years. " +
        "Summary: " + df['bio'].fillna('') + " " +
        "Career: " + df['career_text'].fillna('')
    )
    # Truncate to 512 tokens worth of chars to avoid excessive memory
    df['profile_text'] = df['profile_text'].str[:2000]

    print("Loading sentence-transformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Encoding 100,000 profiles (this may take 10-20 minutes on CPU)...")
    start_time = time.time()
    embeddings = model.encode(
        df['profile_text'].tolist(),
        show_progress_bar=True,
        batch_size=256
    )

    print(f"Encoding finished in {time.time() - start_time:.2f} seconds.")
    print(f"Saving embeddings to {embeddings_file}...")
    np.save(embeddings_file, embeddings)
    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    generate_embeddings(args.data_dir)
