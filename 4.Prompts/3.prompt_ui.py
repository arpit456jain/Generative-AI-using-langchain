from dotenv import load_dotenv
import streamlit as st
import os
import json
import pathlib
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables
load_dotenv()

# Define the path to template.json (relative to this script's location)
template_path = pathlib.Path(__file__).parent / "template.json"

# Check if the file exists before trying to open
if not template_path.exists():
    st.error(f"❌ template.json not found at {template_path}")
    st.stop()

# Load JSON template
with open(template_path, "r", encoding="utf-8") as f:
    template = load_prompt(str(template_path))

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Streamlit UI
st.header('Research Tool')

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

if st.button('Summarize'):
    try:
        chain = template | model
        result = chain.invoke({
            'paper_input': paper_input,
            'style_input': style_input,
            'length_input': length_input
        })
        st.subheader("Summary")
        st.write(result.content if hasattr(result, "content") else result)
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
