from langchain_community.document_loaders import PyPDFLoade

loader = PyPDFLoader("ML_AI.pdf")

docs = loader.load()

print(docs[0].page_content)
print(docs[0].metadata)