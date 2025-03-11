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

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="DocInsight Query System", page_icon="📘", layout="centered")

# Styling
st.markdown('''
<style>
[data-testid="stAppViewContainer"] { background-color: #f0f4f8; color: #333; }
.stTextInput>div>div>input, .stButton>button { border-radius: 10px; border: 1px solid #2563eb; padding: 8px; }
.stButton>button { background-color: #2563eb; color: white; }
.message-container { background-color: #ffffff; border-radius: 10px; padding: 10px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
</style>
''', unsafe_allow_html=True)

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

        splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=200)
        docs = splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
        st.session_state.vectordb = vectordb
        st.success("✅ Document uploaded and processed successfully!")

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
