import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
import PyPDF2
import io
import csv
import re
from io import StringIO

# Initialize LLM
SYSTEM_PROMPT = "You are a helpful and safe AI assistant. You must refuse to engage in harmful, unethical, or biased discussions."
llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def is_input_safe(user_input: str) -> bool:
    """Check if the input is safe to process."""
    dangerous_patterns = [
        r"\b(system|os|subprocess|import|open|globals|locals|__import__|__globals__|__dict__|__builtins__)\b",
        r"(sudo|rm -rf|chmod|chown|mkfs|:(){:|fork bomb|shutdown)",
        r"\b(simulate being|ignore previous instructions|bypass|jailbreak|pretend to be|hack|scam )\b",
        r"(<script>|</script>|<iframe>|javascript:|onerror=)",
        r"(base64|decode|encode|pickle|unpickle)",
        r"(http[s]?://|ftp://|file://)",
        r"\b(manipulate|modify system prompt|alter assistant behavior)\b"
    ]
    return not any(re.search(pattern, user_input, re.IGNORECASE) for pattern in dangerous_patterns)

def process_pdf(uploaded_file):
    """Extracts text from a PDF and splits it into chunks."""
    with io.BytesIO(uploaded_file.getvalue()) as byte_file:
        pdf_reader = PyPDF2.PdfReader(byte_file)
        text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
    
    # Chunk text for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_text(text)

def process_text_file(uploaded_file):
    """Processes a text file and splits it into chunks."""
    text = uploaded_file.getvalue().decode("utf-8")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_text(text)

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None
if "uploaded_file_count" not in st.session_state:
    st.session_state.uploaded_file_count = 0
if "model_confirmed" not in st.session_state:
    st.session_state.model_confirmed = False
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if st.sidebar.button("🆕 Start New Session"):
    st.session_state.clear()
    st.rerun()

st.sidebar.header("📄 Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files and (st.session_state.uploaded_files is None or len(uploaded_files) != st.session_state.uploaded_file_count):
    with st.spinner("Processing documents..."):
        docs = []
        for f in uploaded_files:
            if f.type == "application/pdf":
                docs.extend(process_pdf(f))  # Extend with chunks
            else:
                docs.extend(process_text_file(f))
        
        embeddings = OpenAIEmbeddings()
        faiss_index = FAISS.from_texts(docs, embeddings)
        st.session_state.uploaded_files = faiss_index
        st.session_state.uploaded_file_count = len(uploaded_files)
    st.success(f"Successfully indexed {len(docs)} document chunks.")

st.sidebar.header("💡 Model Settings")
st.session_state.model_choice = st.sidebar.selectbox("Choose Model", ["gpt-3.5-turbo", "gpt-4"], index=0)
st.session_state.model_creativity = st.sidebar.slider("Model Creativity (Temperature)", 0.0, 1.0, 0.7, 0.1)
st.session_state.response_length_words = st.sidebar.slider("Response Length (Words)", 50, 500, 150, 10)

if st.sidebar.button("Confirm Model Settings"):
    st.session_state.model_confirmed = True
    st.success("Model settings confirmed.")

# Displaying conversation history
for message in st.session_state.conversation_history:
    st.chat_message(message["role"]).markdown(message["content"])

if st.session_state.model_confirmed:
    query = st.text_input("Ask a question:", value=st.session_state.user_input)

    # Add a "Send" button for submitting the query
    if st.button("Send Question"):
        if query:
            st.session_state.user_input = ""  # Clear input field
            st.session_state.conversation_history.append({"role": "user", "content": query})

            if is_input_safe(query):
                response_stream = []  # Initialize an empty list to collect response chunks
                if st.session_state.uploaded_files:
                    retriever = st.session_state.uploaded_files.as_retriever(search_kwargs={"k": 2})
                    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", memory=memory)
                    response = qa_chain.run(query)
                else:
                    # Stream response and collect the parts
                    for chunk in llm.stream([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)], 
                                            temperature=st.session_state.model_creativity, max_tokens=512):
                        response_stream.append(str(chunk))  # Append each chunk
                    response = "".join(response_stream)  # Combine all chunks into a single response

                st.session_state.conversation_history.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)
            else:
                response = "⚠️ Your query violates content policies."

                st.session_state.conversation_history.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)
else:
    st.warning("Confirm model settings before asking questions.")

def save_conversation_csv():
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Role", "Message"])
    for msg in st.session_state.conversation_history:
        writer.writerow([msg["role"], msg["content"]])
    return output.getvalue()

st.sidebar.header("💾 Download Conversation")
st.sidebar.download_button("Download CSV", save_conversation_csv(), "conversation.csv", "text/csv")
