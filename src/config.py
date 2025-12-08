# src/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# -----------------------------------------------------
# Embedding Model
# -----------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")

# -----------------------------------------------------
# Qdrant configuration
# -----------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "hendrycks_maths")

# -----------------------------------------------------
# LLM Model (Primary)
# -----------------------------------------------------
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen/qwen3-32b")

# -----------------------------------------------------
# LLM Model (Query Breaker)
# -----------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# -----------------------------------------------------
# LLM Model (Verifier)
# -----------------------------------------------------
GPT_API_KEY = os.getenv("GPT_API_KEY", "")
GPT_MODEL = os.getenv("GPT_MODEL", "openai/gpt-oss-20b")
VERIFIER_TEMPERATURE = float(os.getenv("VERIFIER_TEMPERATURE", 0.0))

# -----------------------------------------------------
# MCP Search Tools
# -----------------------------------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
WIKI_API_KEY = os.getenv("WIKI_API_KEY", "")

# -----------------------------------------------------
# RAG Routing Threshold
# -----------------------------------------------------
RAG_THRESHOLD = float(os.getenv("RAG_THRESHOLD", 0.80))

# -----------------------------------------------------
# Directories
# -----------------------------------------------------
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "qdrant"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

