import os
import requests
from bs4 import BeautifulSoup

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader

# ---------------------- Configuration ----------------------
st.set_page_config(page_title="Chatbot Berita + Upload", layout="wide")
st.title("🤖 Berita & Dokumen Chatbot")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
CHROMA_PATH = "chroma_db"  # Persisted DB directory

# ---------------------- Scrape Function ----------------------
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

# ---------------------- File Loader ----------------------
def load_uploaded_files(uploaded_files):
    all_text = ""
    for file in uploaded_files:
        if file.name.endswith(".txt"):
            text = file.read().decode("utf-8")
            all_text += text + "\n\n"
        elif file.name.endswith(".pdf"):
            with open("temp.pdf", "wb") as f:
                f.write(file.read())
            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()
            all_text += "\n\n".join(doc.page_content for doc in documents)
            os.remove("temp.pdf")
    return all_text

# ---------------------- Vector Store Creation ----------------------
@st.cache_resource(show_spinner="📚 Membuat vector store...")
def build_vectorstore(scraped_text, uploaded_text):
    full_text = scraped_text + "\n\n" + uploaded_text
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = splitter.create_documents([full_text])

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    
    if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
        db = Chroma.from_documents(documents, embeddings, persist_directory=CHROMA_PATH)
        db.persist()
    else:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    return db

# ---------------------- Load Content ----------------------
with st.sidebar:
    st.subheader("🔗 Sumber Data")
    urls_berita = st.text_area("Masukkan URL berita (pisahkan dengan baris)", 
        "https://www.cnbcindonesia.com/news\nhttps://id.wikipedia.org/wiki/Ilmu_ekonomi").splitlines()

    uploaded_files = st.file_uploader("📂 Upload dokumen (PDF/TXT)", accept_multiple_files=True)

    build_button = st.button("🔄 Bangun / Muat Ulang Basis Data")

# Build content on user trigger
if "qa" not in st.session_state or build_button:
    with st.spinner("🔍 Mengambil dan memproses data..."):
        scraped_text = ""
        for url in urls_berita:
            scraped_text += scrape_berita(url) + "\n\n"
        uploaded_text = load_uploaded_files(uploaded_files)
        db = build_vectorstore(scraped_text, uploaded_text)

        llm = OpenAI(api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo-instruct")
        st.session_state.qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

# ---------------------- Chat Interface ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Tanyakan sesuatu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = st.session_state.qa.run(prompt)
            st.markdown(response)
        except Exception as e:
            response = f"⚠️ Terjadi error: {str(e)}"
            st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
