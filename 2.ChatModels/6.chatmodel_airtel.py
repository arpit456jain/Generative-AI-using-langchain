from langchain_perplexity import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
# Use Perplexity API (OpenAI-compatible)
llm = ChatPerplexity(
   model="sonar",
    max_tokens=10
)

# Simple invoke
resp = llm.invoke("What is the capital of India?")
print(resp.content)

# Using with a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{question}")
])

chain = prompt | llm

answer = chain.invoke({"question": "Explain Kafka in simple terms"})
print(answer.content)
