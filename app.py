"""Streamlit UI: upload documents, chat, background re-indexing."""
from datetime import datetime

import streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler

import config
from chatbot import ask
from indexer import sync_index

st.set_page_config(page_title="Dynamic Document Chatbot")


@st.cache_resource
def start_scheduler():
    """Runs once per server process; picks up files dropped into docs/ automatically."""
    sched = BackgroundScheduler()
    sched.add_job(
        sync_index, "interval", minutes=config.REFRESH_MINUTES,
        next_run_time=datetime.now(), max_instances=1,
    )
    sched.start()
    return sched


start_scheduler()

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload PDF / TXT / MD", accept_multiple_files=True,
                               type=[s.strip(".") for s in config.SUPPORTED])
    if uploads and st.button("Index uploads"):
        config.DOCS_DIR.mkdir(exist_ok=True)
        for f in uploads:
            (config.DOCS_DIR / f.name).write_bytes(f.getbuffer())
        with st.spinner("Indexing..."):
            st.success(sync_index())
    st.caption(f"Auto re-index every {config.REFRESH_MINUTES} min")

st.title("Dynamic Document Chatbot")
if "messages" not in st.session_state:
    st.session_state.messages = []

for role, text in st.session_state.messages:
    with st.chat_message("user" if role == "human" else "assistant"):
        st.markdown(text)

if question := st.chat_input("Ask about your documents"):
    with st.chat_message("user"):
        st.markdown(question)
    stream, sources = ask(question, st.session_state.messages)
    with st.chat_message("assistant"):
        answer = st.write_stream(stream)
        if sources:
            with st.expander("Sources"):
                for d in sources:
                    st.caption(f"{d.metadata['source']} · page {d.metadata.get('page', '-')}")
                    st.write(d.page_content[:300] + "...")
    st.session_state.messages += [("human", question), ("ai", answer)]
