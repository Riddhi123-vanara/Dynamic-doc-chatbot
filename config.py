from pathlib import Path

DOCS_DIR = Path("docs")
INDEX_DIR = Path("index_store")
MANIFEST = INDEX_DIR / "manifest.json"
SUPPORTED = {".pdf", ".txt", ".md"}

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4
REFRESH_MINUTES = 10
