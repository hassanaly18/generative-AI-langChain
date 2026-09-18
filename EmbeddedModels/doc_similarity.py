from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "tell me about bumrah"

doc_embedding = embedding.embed_documents(documents)
q_embedding = embedding.embed_query(query)

sim_score = cosine_similarity([q_embedding], doc_embedding)[0]
print(sim_score)
# print(sorted(list(enumerate(sim_score)), key=lambda x:x[1])[-1])

index, score = sorted(list(enumerate(sim_score)), key=lambda x:x[1])[-1]

print("\n"+ query)
print(documents[index])
print("Similarity score is:", score)