import streamlit as st
import os
import time
import tempfile

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# -- 🧠 Sidebar: API Keys and Chat History --
st.sidebar.title("🔐 API Keys & Chat History")

# API Key Inputs
groq_key = st.sidebar.text_input("🧠 GROQ API Key", type="password")
hf_key = st.sidebar.text_input("📚 HuggingFace API Key", type="password")

# Set environment variables
if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key
if hf_key:
    os.environ["HF_API_TOKEN"] = hf_key

# -- 🧠 Initialize chat history --
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# -- 🧾 Main Title --
st.title("📚 Multi-PDF + General Chatbot (RAG + LLM)")

# -- 🎛️ Query Mode --
query_mode = st.radio("Select Query Mode:", ["📄 Ask from PDF", "🌐 General Purpose Q&A"], horizontal=True)

# -- 📄 Upload PDFs --
uploaded_files = st.file_uploader("📂 Upload one or more PDF files", type="pdf", accept_multiple_files=True)

# -- 📌 Prompt Template --
Prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Respond **clearly and concisely**
If you don't know, just say you don't know.
Context: {context}
Question: {input}
""")

# -- 📦 Load & Parse PDFs --
def load_documents_from_files(uploaded_files):
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        all_docs.extend(docs)
        os.remove(tmp_path)
    return all_docs

# -- 🔎 Embedding & Vector DB --
def create_vector_embedding(uploaded_files):
    with st.spinner("🔄 Processing and embedding documents..."):
        documents = load_documents_from_files(uploaded_files)
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = splitter.split_documents(documents)
        st.session_state.final_documents = split_docs
        st.session_state.vectors = FAISS.from_documents(split_docs, st.session_state.embeddings)
        st.success("✅ Vector DB ready!")

# -- 📤 Embed PDFs Button --
if uploaded_files and st.button('🚀 Embed PDFs'):
    create_vector_embedding(uploaded_files)

# -- 💬 Ask Question --
user_prompt = st.text_input("💬 Ask your question:")

# -- 🤖 Answering Logic --
if user_prompt:
    if not groq_key or not hf_key:
        st.warning("🔐 Please enter both GROQ and HuggingFace API keys to proceed.")
        st.stop()

    with st.spinner("🤖 Generating answer..."):
        llm = ChatGroq(model_name="llama3-8b-8192")

        if query_mode == "📄 Ask from PDF":
            if 'vectors' not in st.session_state:
                st.warning("⚠️ Please upload and embed documents first.")
                st.stop()
            doc_chain = create_stuff_documents_chain(llm, Prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, doc_chain)
            start_time = time.process_time()
            response = retrieval_chain.invoke({'input': user_prompt})
            final_answer = response['answer']
        else:
            start_time = time.process_time()
            raw_response = llm.invoke(user_prompt)
            final_answer = raw_response.content if hasattr(raw_response, "content") else str(raw_response)

        response_time = time.process_time() - start_time

        # 🖥️ Display
        st.write(f'⏱️ Response time: {response_time:.2f} seconds')
        st.write('📄 **Answer:**', final_answer)

        # 💾 Save to history
        st.session_state.chat_history.append({
            'question': user_prompt,
            'answer': final_answer,
            'mode': query_mode
        })

# -- 🧠 Sidebar History Preview --
st.sidebar.markdown("### 💬 Chat History")

if st.session_state.chat_history:
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.sidebar.expander(f"{chat['mode'].split()[0]} Q{i+1}: {chat['question'][:50]}..."):
            st.markdown(f"**You:** {chat['question']}")
            st.markdown(f"**Bot:** {chat['answer']}")
