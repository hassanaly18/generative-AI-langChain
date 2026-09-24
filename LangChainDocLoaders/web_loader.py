from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    provider="auto"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

url = "https://www.apple.com/macbook-air/"
loader = WebBaseLoader(url)

docs = loader.load()

prompt = PromptTemplate(
    template="What are the main specs of this device, tell me in a pointed way - \n {data}",
    input_variables=["data"]
)

chain = prompt | model | parser
result = chain.invoke({"data": docs[0].page_content})

print(result)

# print(docs[0].metadata)
# print(len(docs))