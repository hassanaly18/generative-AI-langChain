from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
result = embedding.embed_query("Islamabad is the capital of Pakistan.")

print(str(result))