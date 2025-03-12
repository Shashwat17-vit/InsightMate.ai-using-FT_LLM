import numpy as np
import streamlit as st
import uuid  # To generate unique session IDs
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import tempfile, os
from dotenv import load_dotenv
import sys
import pysqlite3

# Fix NumPy 2.0 issue
if not hasattr(np, 'float_'):  
    np.float_ = np.float64

# Ensure SQLite compatibility for ChromaDB
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Set Streamlit page configuration
st.set_page_config(page_title="📘 DocInsight Query System", page_icon="📘", layout="centered")

st.title("📘 DocInsight Query System")

# ✅ **Step 1: Assign Unique Session ID**
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # Assign a unique session ID

session_id = st.session_state.session_id  # Store the session ID

# ✅ **Step 2: Initialize session-specific storage**
if f"vectordb_{session_id}" not in st.session_state:
    st.session_state[f"vectordb_{session_id}"] = None
    st.session_state[f"memory_{session_id}"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state[f"messages_{session_id}"] = []

# ✅ **Step 3: PDF Upload and Processing (Per User)**
uploaded_file = st.file_uploader("📤 Upload PDF (must contain text)", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        full_text = " ".join(doc.page_content.strip() for doc in documents)

        if not full_text.strip():
            st.error("❌ PDF contains no extractable text. Please upload another document.")
            st.stop()

        # ✅ **Step 4: Process the document (Per User)**
        splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=200)
        docs = splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()

        # ✅ **Step 5: Initialize Chroma (Per User)**
        persist_directory = f"./chroma_db_{session_id}"  # Unique directory per session
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        vectordb.add_documents(docs)

        # Store user-specific vector database
        st.session_state[f"vectordb_{session_id}"] = vectordb
        st.success("✅ Document uploaded and processed successfully! Now you can ask queries!")

    except Exception as e:
        st.error(f"Failed to process the PDF: {str(e)}")
        st.stop()

# ✅ **Step 6: Display User-Specific Conversation History**
st.markdown("### 💬 Conversation")
for msg in st.session_state[f"messages_{session_id}"]:
    role = "**DocInsight:**" if msg["role"] == "ai" else "**You:**"
    st.markdown(f"<div class='message-container'>{role} {msg['content']}</div>", unsafe_allow_html=True)

# ✅ **Step 7: Handle User Queries**
query = st.text_input("Ask DocInsight about your document:", key=f"query_{session_id}")

if st.button("Send") and query:
    if st.session_state[f"vectordb_{session_id}"]:
        qa_chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(model_name="gpt-4-0125-preview"),
            retriever=st.session_state[f"vectordb_{session_id}"].as_retriever(),
            memory=st.session_state[f"memory_{session_id}"]
        )
        response = qa_chain({"question": query})
        
        # Store conversation in session-specific history
        st.session_state[f"messages_{session_id}"].append({"role": "user", "content": query})
        st.session_state[f"messages_{session_id}"].append({"role": "ai", "content": response["answer"]})
        
        st.rerun()
    else:
        st.error("⚠️ Please upload a PDF document first!")

# ✅ **Step 8: Clear Conversation (Only for Current Session)**
if st.button("Clear Conversation"):
    st.session_state[f"vectordb_{session_id}"] = None
    st.session_state[f"memory_{session_id}"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state[f"messages_{session_id}"] = []
    st.rerun()
