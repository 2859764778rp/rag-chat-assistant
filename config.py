import os
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VEC_DIR = DATA_DIR / "vecstore"       
DOC_DIR = BASE_DIR / "documents"         
DB_PATH = DATA_DIR / "crm.db"            
LOG_PATH = DATA_DIR / "app.log"

for _d in (DATA_DIR, VEC_DIR, DOC_DIR):
    _d.mkdir(parents=True, exist_ok=True)


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-chat")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))


STORE_NAME = os.getenv("STORE_NAME", "knowledge")

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.15")) 
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
        ],
    )
    return logging.getLogger("langchain_rag")


logger = configure_logging()
