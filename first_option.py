from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
import requests
from bs4 import BeautifulSoup
import streamlit as st
import os

# ---------------------- Streamlit Setup ----------------------
st.title("Berita Chatbot 📰🤖")

# Set your OpenAI API Key from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# ---------------------- Web Scraper ----------------------
def scrape_berita(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        artikel = soup.find_all('article')
        teks_berita = [item.get_text() for item in artikel]
        return "\n\n".join(teks_berita)
    except Exception as e:
        return f"[ERROR SCRAPING {url}]: {str(e)}"

urls_berita = [
     "https://www.cnbcindonesia.com/news",
    "https://id.wikipedia.org/wiki/Ilmu_ekonomi",
    "https://www.bps.go.id/id",
    "https://www.ekon.go.id/",
    "https://www.cnbcindonesia.com/tag/ekonomi-global",
]

PERSIST_DIR = "chroma_db"

@st.cache_resource(show_spinner="Memproses berita dan membangun database...")
def build_qa_chain():
    all_berita = ""
    for url in urls_berita:
        all_berita += scrape_berita(url) + "\n\n"

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = splitter.create_documents([all_berita])

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # Check if DB already exists
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    else:
        db = Chroma.from_documents(documents, embeddings, persist_directory=PERSIST_DIR)
        # db.persist()

    llm = OpenAI(api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo-instruct")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

    return qa_chain


qa = build_qa_chain()

# ---------------------- Session State ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input and response
if prompt := st.chat_input("Tanyakan sesuatu tentang berita..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Bot response
    with st.chat_message("assistant"):
        try:
            response = qa.invoke(prompt)
            st.markdown(response["result"])
        except Exception as e:
            response = f"⚠️ Terjadi error: {str(e)}"
            st.error(response)
    # st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.messages.append({
    "role": "assistant",
    "content": response["result"],
    "raw_response": response  # opsional, kalau mau simpan semuanya
})
