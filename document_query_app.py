import numpy as np
if not hasattr(np, 'float_'):  # Ensure compatibility with NumPy 2.0
    np.float_ = np.float64

import streamlit as st
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
# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

__import__('pysqlite3')
import sys
import pysqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Set Streamlit page configuration
st.set_page_config(page_title="DocInsight Query System", page_icon="📘", layout="centered")

st.title("📘 DocInsight Query System")

# Initialize or reset session states
def reset_states():
    st.session_state.vectordb = None
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state.messages = []

if 'init' not in st.session_state:
    st.session_state.init = True
    reset_states()

# PDF Uploader
uploaded_file = st.file_uploader("📤 Upload PDF (must contain text)", type=["pdf"], on_change=reset_states)

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

        # Process the document
        splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=200)
        docs = splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()

        # Initialize Chroma with fixed embedding function
        persist_directory = "./chroma_db2"
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        vectordb.add_documents(docs)  # Properly add documents

        # Store in session state
        st.session_state.vectordb = vectordb
        st.success("✅ Document uploaded and processed successfully! Now you can ask queries!")

    except Exception as e:
        st.error(f"Failed to process the PDF: {str(e)}")
        st.stop()

# Display conversation history
st.markdown("### 💬 Conversation")
for msg in st.session_state.messages:
    role = "**DocInsight:**" if msg["role"] == "ai" else "**You:**"
    st.markdown(f"<div class='message-container'>{role} {msg['content']}</div>", unsafe_allow_html=True)

# User input and response handling
query = st.text_input("Ask DocInsight about your document:", key="query")
if st.button("Send") and query:
    if st.session_state.vectordb:
        qa_chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(model_name="gpt-4-0125-preview"),
            retriever=st.session_state.vectordb.as_retriever(),
            memory=st.session_state.memory
        )
        response = qa_chain({"question": query})
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "ai", "content": response["answer"]})
        st.rerun()
    else:
        st.error("⚠️ Please upload a PDF document first!")

# Clear conversation button
if st.button("Clear Conversation"):
    reset_states()
    st.rerun()
