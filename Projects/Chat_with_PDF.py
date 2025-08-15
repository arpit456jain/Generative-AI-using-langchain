import os
import io
import tempfile
import textwrap
import pathlib
from typing import List

import streamlit as st
from dotenv import load_dotenv

# LangChain + community integrations
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


# ----------------------------
# Page & Globals
# ----------------------------
st.set_page_config(page_title="Chat with your PDFs (RAG)", page_icon="📄", layout="wide")
load_dotenv()

# ----------------------------
# Helper: get secret safely
# ----------------------------
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
chat_model = ChatHuggingFace(llm=llm)

# ----------------------------
# Session State
# ----------------------------
if "docs" not in st.session_state:
    st.session_state.docs = []  # LangChain Documents (post-split)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "history" not in st.session_state:
    st.session_state.history = []  # [(user, assistant)]
if "summary" not in st.session_state:
    st.session_state.summary = ""

# ----------------------------
# UI: Sidebar
# ----------------------------
with st.sidebar:
    st.header("📎 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drop 1 or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Build / Rebuild Index", type="primary"):
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            with st.spinner("Reading PDFs and building the vector index..."):
                # Load & combine pages from all PDFs
                raw_docs = []
                for up in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(up.read())
                        tmp.flush()
                        loader = PyPDFLoader(tmp.name)
                        pages = loader.load()
                        raw_docs.extend(pages)

                # Split docs into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1200,
                    chunk_overlap=200,
                    length_function=len,
                )
                split_docs = splitter.split_documents(raw_docs)
                st.session_state.docs = split_docs

                # Build vectorstore with HF embeddings (no API key needed)
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vs = FAISS.from_documents(split_docs, embeddings)
                st.session_state.vectorstore = vs

            st.success(f"Indexed {len(split_docs)} chunks from {len(uploaded_files)} PDF(s).")

    st.markdown("---")
    if st.button("Clear Chat"):
        st.session_state.history = []
        st.success("Cleared chat history.")

    st.markdown("**Tip:** After rebuilding the index, ask questions in the main panel.")

# ----------------------------
# Main: Header
# ----------------------------
st.title("📄 Chat with your PDFs")
st.caption("Upload PDFs, get a summary, and ask grounded questions with Retrieval-Augmented Generation (RAG).")

index_ready = st.session_state.vectorstore is not None and len(st.session_state.docs) > 0

# ----------------------------
# Summarization (Map-Reduce Lite)
# ----------------------------
st.subheader("📝 Summarize")
col1, col2 = st.columns([3, 1])

with col1:
    summary_style = st.selectbox(
        "Style",
        ["Concise", "Detailed", "Bullet Points", "Beginner-Friendly", "Executive Summary"]
    )
with col2:
    max_chunks_to_summarize = st.number_input(
        "Chunks to use",
        min_value=3,
        max_value=100,
        value=15,
        step=1,
        help="Use more chunks for longer summaries; costs more tokens."
    )

def summarize_chunks(chunks_text: List[str], style: str) -> str:
    """Summarize multiple chunks then reduce into a final summary."""
    per_chunk_summaries = []
    instruction = {
        "Concise": "Write a concise summary (4-8 sentences).",
        "Detailed": "Write a thorough, well-structured summary with sections.",
        "Bullet Points": "Summarize as clear bullet points.",
        "Beginner-Friendly": "Explain like I'm new to the topic; avoid heavy jargon.",
        "Executive Summary": "Write an executive summary focusing on key findings, implications, and risks."
    }[style]

    # Map step
    for i, chunk in enumerate(chunks_text, start=1):
        prompt = (
            f"You are a helpful assistant. {instruction}\n\n"
            f"TEXT CHUNK {i}:\n{chunk}\n\n"
            "Summary:"
        )
        res = chat_model.invoke(prompt)
        per_chunk_summaries.append(res.content if hasattr(res, "content") else str(res))

    # Reduce step
    reduce_prompt = (
        f"Combine the following partial summaries into one {style.lower()} summary. "
        f"Avoid redundancy; ensure coherence, and keep it self-contained.\n\n"
        f"PARTIAL SUMMARIES:\n{chr(10).join(per_chunk_summaries)}\n\n"
        "Final Summary:"
    )
    final = chat_model.invoke(reduce_prompt)
    return final.content if hasattr(final, "content") else str(final)

if st.button("Generate Summary", disabled=not index_ready):
    if not index_ready:
        st.warning("Please upload PDFs and build the index first.")
    else:
        with st.spinner("Summarizing..."):
            # Use first N chunks for a representative summary
            chunks = [d.page_content for d in st.session_state.docs[: max_chunks_to_summarize]]
            st.session_state.summary = summarize_chunks(chunks, summary_style)
        st.success("Summary generated!")

if st.session_state.summary:
    st.markdown("#### Summary")
    st.write(st.session_state.summary)
    st.download_button(
        "Download Summary (.txt)",
        data=st.session_state.summary,
        file_name="summary.txt",
        mime="text/plain",
    )

st.markdown("---")

# ----------------------------
# Chat with Retrieval
# ----------------------------
st.subheader("💬 Ask Questions about your PDFs")

if not index_ready:
    st.info("Upload PDFs and click **Build / Rebuild Index** in the sidebar to start chatting.")
else:
    # Show prior chat
    for u, a in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(u)
        with st.chat_message("assistant"):
            st.markdown(a)

    # Chat input
    user_q = st.chat_input("Ask a question grounded in your PDFs…")
    if user_q:
        # Retrieve top-k relevant chunks
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(user_q)

        # Build context
        context_blocks = []
        for idx, d in enumerate(docs, start=1):
            src = d.metadata.get("source", "PDF")
            page_num = d.metadata.get("page", "N/A")
            context_blocks.append(f"[Chunk {idx} | {src} | page {page_num}]\n{d.page_content}")

        context_text = "\n\n".join(context_blocks)
        system_instructions = (
            "You are a careful assistant. Answer the user's question using ONLY the provided context. "
            "If the answer is not in the context, say you don't know."
        )
        prompt = (
            f"{system_instructions}\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"QUESTION: {user_q}\n\n"
            "Answer:"
        )

        with st.spinner("Thinking..."):
            result = chat_model.invoke(prompt)
            answer = result.content if hasattr(result, "content") else str(result)

        # Save history
        st.session_state.history.append((user_q, answer))

        # Display newest exchange
        with st.chat_message("user"):
            st.markdown(user_q)
        with st.chat_message("assistant"):
            st.markdown(answer)

        # Expandable: show sources/chunks used
        with st.expander("Show retrieved context"):
            for i, d in enumerate(docs, start=1):
                src = d.metadata.get("source", "PDF")
                page_num = d.metadata.get("page", "N/A")
                st.markdown(f"**Chunk {i}** — *{src}*, page **{page_num}**")
                st.write(d.page_content)
                st.markdown("---")
