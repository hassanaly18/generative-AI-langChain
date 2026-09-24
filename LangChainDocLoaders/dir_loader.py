from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

loader = DirectoryLoader(
    path="books",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load() #load() vs lazy_load()

# print(len(docs))
# print(docs[0].metadata)
# print(docs[1].metadata)
# print(docs[-1].metadata)

for doc in docs:
    print(doc.metadata)