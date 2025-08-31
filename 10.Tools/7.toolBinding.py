# tool create
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv


@tool
def multiply(a: int, b: int) -> int:
  """Given 2 numbers a and b this tool returns their product"""
  return a * b

# print(multiply.invoke({'a':3, 'b':4}))
# print(multiply.name)
# print(multiply.description)
# print(multiply.args)

# tool binding
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# result = model.invoke("What is the capital of India")

# not ever llm has the capability to bind with tool
model_with_tools = model.bind_tools([multiply])
# result = model_with_tools.invoke('Hi how are you')
# print(result.content)

query = HumanMessage('can you multiply 3 with 1000')
messages = [query]
result = model_with_tools.invoke(messages)
messages.append(result)
print(messages)

# tool_result = multiply.invoke(result.tool_calls[0])
# messages.append(tool_result)

# result = model_with_tools.invoke(messages).content
# print(result.content)