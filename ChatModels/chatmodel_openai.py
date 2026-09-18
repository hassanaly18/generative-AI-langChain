from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    provider="auto",
    temperature=1
)

model = ChatHuggingFace(llm=llm)
# result = model.invoke("What is the capital of Pakistan?")
result = model.invoke("who won the 2026 FIFA world cup")

# print(result)
print(result.content)



# from langchain_huggingface import ChatHuggingFace
# from dotenv import load_dotenv

# load_dotenv()

# model = ChatHuggingFace(repo_id="HuggingFaceH4/zephyr-7b-beta")
# result = model.invoke("What is the capital of Pakistan?")

# print(result)