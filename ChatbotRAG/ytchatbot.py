from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

video_id = "Gfr50f6ZBvo"

#Indexing
try:
    ytt_api = YouTubeTranscriptApi()          
    transcript_list = ytt_api.fetch(video_id, languages=["en"])

    # Flatten it to plain text
    transcript = " ".join(snippet.text for snippet in transcript_list)
    #print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

print(len(chunks))
# print(chunks[0])

embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

print()
vectorstore.index_to_docstore_id

#Retrieval
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
retriever.invoke("what is deepmind") #this will fetch the relevant docs from all the docs

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    provider="auto",
    max_new_tokens = 512
   # temperature=1
)

model = ChatHuggingFace(llm=llm)

#Augmentation
prompt = PromptTemplate(
    template="""
    You are a helpful assistant. 
    Answer ONLY from the provided transcript context.if the context is insufficient, just say you dont know.
    {context}
    Question: {question}
    """,
    input_variables=["context", "question"]
)

question = "is the topic of aliens discussed in this video? if yes what was discussed?"
retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text, "question": question})
# print(final_prompt)

#Generation
answer = model.invoke(final_prompt)
print(answer.content)