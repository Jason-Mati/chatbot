from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import os


st.set_page_config(
    page_title="KB HRD 챗봇",
    page_icon="📃",
)

# Path to the pre-designated file
PRE_REGISTERED_FILE_PATH = './files/hrdrule.pdf'  # Change this to your file's path

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

@st.cache_data(show_spinner="학습정보를 가져오고 있어요...")
def embed_file(file_path, api_key):
    with open(file_path, "rb") as file:
        file_content = file.read()

    # Process and embed the file content
    cache_dir = LocalFileStore(f"./.cache/embeddings/{os.path.basename(file_path)}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)  # Use the user's API key here
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    if "messages" in st.session_state:
        for message in st.session_state["messages"]:
            send_message(
                message["message"],
                message["role"],
                save=False,
            )

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. Do not learn or retain any information from these documents beyond this session. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

st.title("KB HRD 챗봇")

st.markdown(
    """
안녕하세요, 저는 KB손해보험 인재개발파트에서 개발한 "KB HRD 챗봇"입니다.

OpenAI의 GPT API, Streamlit, Langchain 등을 활용하여 만들어졌으며, HRD 제도집을 열심히 학습했답니다.

궁금하신 점을 물어봐주시면, 학습한 범위 내에서 일목요연하게 설명해드릴게요 ^^
"""
)

# Sidebar: User input for OpenAI API Key
with st.sidebar:
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

# Check if the API key is provided
if openai_api_key:
    st.session_state["openai_api_key"] = openai_api_key

    # Load the pre-registered file
    if os.path.exists(PRE_REGISTERED_FILE_PATH):
        retriever = embed_file(PRE_REGISTERED_FILE_PATH, st.session_state["openai_api_key"])
        send_message("무엇이 궁금하신가요?", "ai", save=False)
        paint_history()
        message = st.chat_input("궁금하신 점을 여기에 적어주세요...")
        if message:
            send_message(message, "human")
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | ChatOpenAI(temperature=0.1, streaming=True, callbacks=[ChatCallbackHandler()], openai_api_key=st.session_state["openai_api_key"])  # Pass the user's API key
            )
            with st.chat_message("ai"):
                chain.invoke(message)
    else:
        st.error(f"The file at {PRE_REGISTERED_FILE_PATH} does not exist.")
else:
    st.error("Please enter your OpenAI API Key in the sidebar to start.")

###