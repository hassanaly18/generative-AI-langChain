from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    provider="auto"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

loader = TextLoader("cricket.txt", encoding="utf-8")

docs = loader.load()

prompt = PromptTemplate(
    template="Write a single line summary for the following poem. \n {poem}",
    input_variables=["poem"]
)

chain = prompt | model | parser
result = chain.invoke({"poem": docs[0].page_content})

print(result)

# print(type(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)
warnings.filterwarnings("ignore", category=DeprecationWarning)
