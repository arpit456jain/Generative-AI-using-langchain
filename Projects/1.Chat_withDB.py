import os
import pandas as pd
from sqlalchemy import create_engine
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
import streamlit as st
from dotenv import load_dotenv
from typing import List, Optional, Any
from langchain_perplexity import ChatPerplexity
from langchain.llms.base import LLM
from pydantic import Field
from langchain_core.prompts import PromptTemplate
# Load env variables
load_dotenv()

DB_URL = "postgresql://postgres:1234@localhost:5432/demo"

# --------------------
# 1️⃣ LOAD DATA FROM POSTGRES
# --------------------
def load_data_from_db(query):
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

df = load_data_from_db("SELECT * FROM student")

# --------------------
# 2️⃣ PREPARE DOCUMENTS
# --------------------
text_data = "\n".join(df.astype(str).apply(lambda row: " | ".join(row), axis=1))
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = [Document(page_content=chunk) for chunk in splitter.split_text(text_data)]

# --------------------
# 3️⃣ CREATE VECTOR STORE
# --------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={"device": "cpu"}  # Force CPU
                                   )
vectorstore = FAISS.from_documents(docs, embeddings)

# --------------------
# 4️⃣ LLM SETUP (NEW API)
# --------------------

    
llm = ChatPerplexity(
   model="sonar",
   
    max_tokens=50
)

# Strict RAG prompt
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant. 
Answer ONLY based on the following context from the database. 
If the answer is not in the context, reply strictly with: "I don’t know".

Context:
{context}

Question:
{question}

Answer:
"""
)
# 5️⃣ RAG Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT}
)

# --------------------
# 5️⃣ STREAMLIT UI
# --------------------
st.title("💬 Chat with PostgreSQL (RAG + Hugging Face)")

user_query = st.text_input("Ask a question about your database:")
# user_query = "Tell me the name of students having marks more than 20"
# result = qa.invoke({"query": user_query})   
# print(result)

if st.button("Ask") and user_query:
    result = qa.invoke({"query": user_query})  # ✅ use invoke() instead of __call__
    st.subheader("Answer:")
    st.write(result["result"])

    # with st.expander("See sources"):
    #     for doc in result["source_documents"]:
    #         st.write(doc.page_content)
