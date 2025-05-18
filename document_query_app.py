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
from langchain.schema import AIMessage, HumanMessage, SystemMessage  # ✅ FIXED: Use correct message format
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
st.set_page_config(page_title="📘 InsightMate.ai", page_icon="📘", layout="centered")

# ✅ **New Content About the Application**
st.markdown("""
# 📘 InsightMate.ai  
#### Built by **Shashwat Negi**  
> 💡 Know more about him by asking questions in the textbox below.  
> 🚀 The chatbot uses a fine-tuned LLM model and may produce **inaccurate** results.  
> 📌 The **best way to know about me** is by connecting on LinkedIn:  

[![Connect on LinkedIn](https://img.shields.io/badge/Connect%20on%20LinkedIn-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/shashwat-negi3/)

---
📂 **Upload a PDF** to receive **more accurate and faster responses** based on the document's content.
""", unsafe_allow_html=True)

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
        st.success("✅ Document uploaded and processed successfully! Now you can ask queries based on the document!")

    except Exception as e:
        st.error(f"Failed to process the PDF: {str(e)}")
        st.stop()

# ✅ **Step 6: Display User-Specific Conversation History**
st.markdown("### 💬 Conversation")
for msg in st.session_state[f"messages_{session_id}"]:
    role = "**DocInsight:**" if msg["role"] == "ai" else "**You:**"
    st.markdown(f"<div class='message-container'>{role} {msg['content']}</div>", unsafe_allow_html=True)

# ✅ **Step 7: Handle User Queries (Two Modes)**
query = st.text_input("Ask a question about Shashwat or uploaded document:", key=f"query_{session_id}")

if st.button("Send") and query:
    if st.session_state[f"vectordb_{session_id}"]:
        # If document is uploaded, use retrieval-based answers
        qa_chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(model_name="ft:gpt-3.5-turbo-0125:personal:negi3:BD1U9AZL"),  # ✅ Using fine-tuned model
            retriever=st.session_state[f"vectordb_{session_id}"].as_retriever(),
            memory=st.session_state[f"memory_{session_id}"]
        )
        response = qa_chain({"question": query})  # Use document retrieval

    else:
        # If no document is uploaded, answer general questions about Shashwat
        chat_model = ChatOpenAI(model_name="ft:gpt-3.5-turbo-0125:personal:negi3:BD1U9AZL")
        response = chat_model.predict_messages([
            SystemMessage(content="You are a helpful assistant. Answer user queries based on Shashwat Negi's skills, experience, and projects."),  # ✅ FIXED SYSTEM MESSAGE
            HumanMessage(content=query)  # ✅ FIXED HUMAN MESSAGE
        ])
        response = {"answer": response.content}  # Extract the response text

    # Store conversation in session-specific history
    st.session_state[f"messages_{session_id}"].append({"role": "user", "content": query})
    st.session_state[f"messages_{session_id}"].append({"role": "ai", "content": response["answer"]})

    st.rerun()

# ✅ **Step 8: Clear Conversation (Only for Current Session)**
if st.button("Clear Conversation"):
    st.session_state[f"vectordb_{session_id}"] = None
    st.session_state[f"memory_{session_id}"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state[f"messages_{session_id}"] = []
    st.rerun()
