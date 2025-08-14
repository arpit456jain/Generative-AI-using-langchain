from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)

#temperature decides the creativity 
# lower value --> deterministic
# higher value --> creative

result = model.invoke("Write a 5 line poem on cricket")

print(result.content)