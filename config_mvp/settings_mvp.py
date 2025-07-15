import os
from dotenv import load_dotenv

# Determine the path to the .env file (assuming it's in the project root)
# settings_mvp.py is in config_mvp/, so to reach the project root, we go up one level.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dotenv_path = os.path.join(PROJECT_ROOT, '.env')

# Load the .env file
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, override=True)
    print(f"Loaded .env file from: {dotenv_path} (overriding existing shell variables if any)")
else:
    print(f"Warning: .env file not found at {dotenv_path}. Using default values or environment variables if set elsewhere.")

# --- General API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Embedding Model Configuration (e.g., LM Studio) ---
EMBEDDING_MODEL_API_URL = os.getenv("EMBEDDING_MODEL_API_URL", "http://10.2.20.2:1235/v1/embeddings")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-aiteamvn-vietnamese_embedding")

# --- Main LLM Configuration (e.g., LM Studio) ---
# These were in the original .env example, adding them here for completeness from MVP plan.
MAIN_LLM_API_URL = os.getenv("MAIN_LLM_API_URL", "http://localhost:1234/v1") 
TEXT_TO_SQL_LLM_API_URL = os.getenv("TEXT_TO_SQL_LLM_API_URL", "http://localhost:1234/v1")


# --- Database Configuration ---
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://huygdo@localhost:5432/shbfc_dwh")

# --- Vector Database Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "policy_documents_mvp")


# --- Sanity Checks & Logging (Optional but Recommended) ---
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY is not set. This might be required for some functionalities.")

# You can add more checks here, for example, to ensure critical URLs are well-formed.

# Example of how to use these settings in other modules:
# from config_mvp.settings_mvp import QDRANT_URL, POSTGRES_DSN
# print(f"Qdrant is running at: {QDRANT_URL}")
