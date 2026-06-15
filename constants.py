# constants.py
# Shared constants used by both rank.py and app.py.
# Centralised here to prevent the two files from drifting apart.

CONSULTING_FIRMS = [
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini',
    'hcl', 'tech mahindra', 'mindtree', 'hexaware', 'mphasis', 'ltimindtree'
]

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

PROFICIENCY_MULT = {
    'beginner': 0.3,
    'intermediate': 0.6,
    'advanced': 0.85,
    'expert': 1.0,
}

# Maximum theoretical raw score before behavioural multiplier scaling.
# Used to normalise final scores to 0–100.
MAX_RAW_SCORE = 109.25
