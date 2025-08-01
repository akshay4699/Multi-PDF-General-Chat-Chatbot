# 📚 Multi-PDF + General Chatbot (RAG + LLM)

A Streamlit chatbot app that lets you:
- 💬 Ask questions from uploaded PDFs using Retrieval-Augmented Generation (RAG)
- 🌐 Switch to general-purpose Q&A using LLaMA 3 via GROQ
- 📁 Embed multiple PDFs into FAISS vectorstore
- 🧠 View complete chat history in the sidebar

---

## 🔗 Live Demo

👉 [Launch the App](https://multi-pdf-general-chat-chatbot-hazehxn8cqwakgux7jtv5k.streamlit.app/)

---

## 🚀 Features

✅ Multi-PDF Upload and Parsing  
✅ HuggingFace Sentence Transformers for Embeddings  
✅ FAISS VectorStore for Retrieval  
✅ GROQ LLaMA3-based Answering (RAG + LLM)  
✅ Toggle between PDF and General Q&A  
✅ Chat History Viewer in Sidebar  
✅ Streamlit UI with clean layout  

---

## 🛠️ Setup Instructions

### 1. 🔃 Clone the Repository
```bash
git clone https://github.com/akshay4699/Multi-PDF-General-Chat-Chatbot.git
cd Multi-PDF-General-Chat-Chatbot

### 2. 🧪 Create Virtual Environment (optional but recommended)
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Mac/Linux

### 3. 📦 Install Dependencies
```bash
pip install -r requirements.txt

### 4. 🔐 Set Your API Keys
You can enter them in the sidebar UI when the app launches:

GROQ API Key: [Get it here] (https://console.groq.com/)

HuggingFace Token: [Get it here] (https://huggingface.co/settings/tokens)

### 5. ▶️ Run the App
streamlit run app.py

