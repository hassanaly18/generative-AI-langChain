from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    provider="auto"
)

prompt = PromptTemplate(
    template="suggest a catchy blog title about {topic}",
    input_variables=["topic"]
)

chain = LLMChain(llm=llm, prompt=prompt)

topic = "cricket"
output = chain.run(topic)

print(output)