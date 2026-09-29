import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


def extract_video_id(url):
    """Pull the video ID out of common YouTube URL formats."""
    parsed = urlparse(url.strip())
    if parsed.hostname in ("youtu.be", "www.youtu.be"):
        return parsed.path.lstrip("/")
    if parsed.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        if parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
            return parsed.path.split("/")[2]
    return None


#Indexing + Retrieval (cached per video, so asking more questions is instant)
@st.cache_resource(show_spinner=False)
def build_retriever(video_id):
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id, languages=["en"])

    # Flatten it to plain text
    transcript = " ".join(snippet.text for snippet in transcript_list)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})


# Model (loaded once)
@st.cache_resource(show_spinner=False)
def build_model():
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        provider="auto",
        max_new_tokens=512,
    )
    return ChatHuggingFace(llm=llm)

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

parser = StrOutputParser()


def build_chain(retriever, model):
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })
    return parallel_chain | prompt | model | parser


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="YouTube Chatbot", page_icon="🎥")
st.title("🎥 Chat with a YouTube Video")

url = st.text_input("YouTube link", placeholder="https://www.youtube.com/watch?v=...")
question = st.text_area("Your question", placeholder="Can you summarize the video in one sentence?")

if st.button("Ask", type="primary"):
    if not url or not question:
        st.warning("Please enter both a YouTube link and a question.")
    else:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Couldn't find a video ID in that link. Please check the URL.")
        else:
            try:
                with st.spinner("Processing the video transcript..."):
                    retriever = build_retriever(video_id)
                model = build_model()
                main_chain = build_chain(retriever, model)

                with st.spinner("Thinking..."):
                    answer = main_chain.invoke(question)

                st.subheader("Answer")
                st.write(answer)

            except TranscriptsDisabled:
                st.error("No captions available for this video.")
            except Exception as e:
                st.error(f"Something went wrong: {e}")