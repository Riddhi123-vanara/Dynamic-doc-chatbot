"""Retrieval + answer generation."""
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

import config
from indexer import embeddings

_cache = {"mtime": None, "store": None}

SYSTEM = (
    "You answer questions using ONLY the provided context from the user's documents. "
    "If the answer is not in the context, say you couldn't find it in the documents. "
    "Cite sources as [filename, page] where available."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        MessagesPlaceholder("history"),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)
llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0, streaming=True)
chain = prompt | llm | StrOutputParser()


def get_store():
    """Reload the index from disk only when the scheduled job has updated it."""
    path = config.INDEX_DIR / "index.faiss"
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    if _cache["mtime"] != mtime:
        _cache["store"] = FAISS.load_local(
            str(config.INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
        _cache["mtime"] = mtime
    return _cache["store"]


def ask(question: str, history: list[tuple[str, str]]):
    """Returns (token_stream, source_docs). history = [("human"|"ai", text), ...]"""
    store = get_store()
    if store is None:
        return iter(["No documents indexed yet. Upload a file first."]), []
    docs = store.similarity_search(question, k=config.TOP_K)
    context = "\n\n".join(
        f"[{d.metadata['source']}, p.{d.metadata.get('page', '-')}]\n{d.page_content}" for d in docs
    )
    stream = chain.stream({"context": context, "question": question, "history": history[-6:]})
    return stream, docs
