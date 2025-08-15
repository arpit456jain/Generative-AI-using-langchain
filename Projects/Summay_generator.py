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
st.header('Summary Generator Tool')
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