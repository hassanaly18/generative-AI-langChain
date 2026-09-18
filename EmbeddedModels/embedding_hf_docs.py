from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2") #384 dimensional

documents = [
    "Islamabad is the capital of Pakistan.",
    "Lahore is the provincial capital of Punjab",
    "Paris is the capital of France"
]
result = embedding.embed_documents(documents)

print(str(result))