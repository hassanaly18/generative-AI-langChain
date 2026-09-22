from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    provider="auto"
)

model = ChatHuggingFace(llm=llm)

#1st prompt
template1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=["topic"] 
)

#2nd prompt 
template2 = PromptTemplate(
    template="write a 5 line summary on the following text. \n {text}",
    input_variables=["text"] 
)

prompt1 = template1.invoke({"topic": "Black hole"})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({"text": result1.content})
result2 = model.invoke(prompt2)

print(result2.content)