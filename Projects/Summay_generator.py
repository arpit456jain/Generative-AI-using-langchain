from dotenv import load_dotenv
import streamlit as st
import os
import json
import pathlib
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
import tempfile

# Load environment variables
load_dotenv()

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# loader = PyPDFLoader('dl-curriculum.pdf')
# Streamlit UI
st.header('📄 PDF Summary & Chat Tool')
pdf_text = ""  # Store PDF text globally in app
# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        # Combine text from all pages
        pdf_text = "\n".join([page.page_content for page in pages])
        st.success(f"✅ Loaded {len(pages)} pages from PDF.")
    
if st.button("Summarize"):
        if len(pdf_text) > 4000:
            pdf_text = pdf_text[:4000]  # Keep within context limit
        prompt = f"Summarize the following text in a clear, concise way:\n\n{pdf_text}"
        result = model.invoke(prompt)
        st.subheader("📌 Summary")
        st.write(result.content if hasattr(result, "content") else result)


# Chat with PDF
if pdf_text:
    st.subheader("💬 Ask Questions About the PDF")
    user_question = st.text_input("Enter your question:")
    if st.button("Ask"):
        if user_question.strip():
            text_to_use = pdf_text[:4000] if len(pdf_text) > 4000 else pdf_text
            chat_prompt = f"Answer the following question based ONLY on the provided PDF content.\n\nPDF Content:\n{text_to_use}\n\nQuestion: {user_question}"
            chat_result = model.invoke(chat_prompt)
            st.subheader("🤖 Answer")
            st.write(chat_result.content if hasattr(chat_result, "content") else chat_result)
        else:
            st.warning("Please type a question.")