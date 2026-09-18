# from langchain_openai import OpenAI
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# llm = OpenAI(model="gpt-3.5-turbo-instruct")

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct", task="text-generation")
result = llm.invoke("What is the capital of Pakistan")
print(result)