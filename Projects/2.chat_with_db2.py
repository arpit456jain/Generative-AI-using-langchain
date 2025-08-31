import os
import sys
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import PGVector
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

# Load config
load_dotenv()
DB_URL = os.getenv("DB_URL")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not DB_URL or not HF_TOKEN:
    print("❌ DB_URL or HUGGINGFACEHUB_API_TOKEN missing in .env")
    sys.exit(1)

# 1️⃣ Load table
TABLE = "student"
engine = create_engine(DB_URL)
df = pd.read_sql(f"SELECT * FROM {TABLE}", engine)

if df.empty:
    print(f"⚠️ Table '{TABLE}' is empty.")
    sys.exit(0)

print(f"✅ Loaded {len(df)} rows from '{TABLE}'")

# 2️⃣ Build documents
docs = [
    Document(page_content=" | ".join([f"{col}: {row[col]}" for col in df.columns]))
    for _, row in df.iterrows()
]

# 3️⃣ Embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

vectorstore = PGVector.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="student_docs",
    connection_string=DB_URL
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("🧠 Vectorstore ready")

# 4️⃣ Hugging Face LLM
llm = HuggingFaceEndpoint(
     repo_id="google/flan-t5-small",
    task="text-generation",
    temperature=0.1,
    max_new_tokens=64,
    huggingfacehub_api_token=HF_TOKEN
)

# 5️⃣ Simple query test
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

query = "max score"
result = qa.invoke({"query": query})
print("\n🧾 Answer:\n", result["result"])
