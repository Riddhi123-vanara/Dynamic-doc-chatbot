# Dynamic Document Chatbot

Python · OpenAI API · LangChain · FAISS

Chat with your own documents. Retrieval runs over a FAISS index; a background job
keeps the index in sync with the `docs/` folder.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...        # Windows: set OPENAI_API_KEY=sk-...
streamlit run app.py
```
Drop files into `docs/` (or upload in the sidebar). New or changed files are indexed
automatically every `REFRESH_MINUTES`; deleted files are removed from the index.
`python indexer.py` runs a one-off sync from the CLI.

## How it works
- `indexer.py` – loads PDF/TXT/MD, splits with `RecursiveCharacterTextSplitter`,
  attaches metadata (source, chunk_id, file_hash, indexed_at) per chunk, and embeds into FAISS.
  A manifest of file hashes makes updates incremental (no full rebuilds).
- `chatbot.py` – top-k similarity search, grounded prompt, streamed answer with sources.
- `app.py` – Streamlit chat UI + APScheduler job for scheduled indexing.
