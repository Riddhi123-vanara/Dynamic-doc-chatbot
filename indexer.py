"""Build and incrementally update the FAISS index."""
import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

_lock = threading.Lock()
embeddings = OpenAIEmbeddings(model=config.EMBED_MODEL)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],  # prefer paragraph, then sentence breaks
)


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_and_chunk(path: Path, digest: str):
    loader = PyPDFLoader(str(path)) if path.suffix.lower() == ".pdf" else TextLoader(str(path), encoding="utf-8")
    chunks = splitter.split_documents(loader.load())
    now = datetime.now(timezone.utc).isoformat()
    for i, chunk in enumerate(chunks):
        chunk.metadata.update(
            {"source": path.name, "chunk_id": i, "file_hash": digest, "indexed_at": now}
        )
    return chunks


def _load_store():
    if (config.INDEX_DIR / "index.faiss").exists():
        return FAISS.load_local(
            str(config.INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
    return None


def sync_index() -> dict:
    """Add new/changed files, drop deleted/changed ones. Safe to call repeatedly."""
    with _lock:
        config.INDEX_DIR.mkdir(exist_ok=True)
        config.DOCS_DIR.mkdir(exist_ok=True)
        manifest = json.loads(config.MANIFEST.read_text()) if config.MANIFEST.exists() else {}
        store = _load_store()

        current = {
            p.name: p for p in config.DOCS_DIR.iterdir() if p.suffix.lower() in config.SUPPORTED
        }
        digests = {name: file_hash(p) for name, p in current.items()}

        # 1. remove vectors for deleted or modified files
        stale_ids, removed = [], 0
        for name in list(manifest):
            if name not in current or manifest[name]["hash"] != digests[name]:
                stale_ids += manifest.pop(name)["ids"]
                removed += 1
        if stale_ids and store:
            store.delete(stale_ids)

        # 2. embed and add new or modified files
        added = 0
        for name, path in current.items():
            if name in manifest:
                continue
            chunks = load_and_chunk(path, digests[name])
            if not chunks:
                continue
            ids = [f"{name}::{i}" for i in range(len(chunks))]
            if store is None:
                store = FAISS.from_documents(chunks, embeddings, ids=ids)
            else:
                store.add_documents(chunks, ids=ids)
            manifest[name] = {"hash": digests[name], "ids": ids}
            added += 1

        if store:
            store.save_local(str(config.INDEX_DIR))
        config.MANIFEST.write_text(json.dumps(manifest, indent=2))
        return {"added": added, "removed": removed, "total_files": len(manifest)}


if __name__ == "__main__":
    print(sync_index())
