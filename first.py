from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
load_dotenv()

topic = "why don't skeletons fight? Because they don't have guts!"

summary_template = "Tell me a joke about {topic}"
promt_template = PromptTemplate(input_variables=["topic"], template=summary_template)

# Fill the template
filled_prompt = promt_template.format(topic=topic)

# Initialize LLM
# llm = ChatOpenAI(model="gpt-3.5-turbo")  # Make sure OPENAI_API_KEY is in your .env
llm = ChatOllama(model="llama3") 
# Get response
response = llm.invoke(filled_prompt)

print("Prompt:", filled_prompt)
print("LLM Response:", response.content)
