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


st.title("🤖 AI Chatbot - Ask Me Anything!")

# Initialize LLM
SYSTEM_PROMPT = "You are a helpful and safe AI assistant. You must refuse to engage in harmful, unethical, or biased discussions."
llm = ChatOpenAI(model="gpt-3.5-turbo")

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


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
        # Process new documents and append them to existing docs
        for f in uploaded_files:
            if f.type == "application/pdf":
                docs.extend(process_pdf(f))  # Extend with chunks
            else:
                docs.extend(process_text_file(f))
        
        # If there's already an existing FAISS index, append to it
        if st.session_state.uploaded_files:
            current_index = st.session_state.uploaded_files
            current_docs = current_index.docstore._docs  # Extract existing docs from FAISS
            docs.extend(current_docs)  # Append the old docs to the new ones
        
        # Rebuild FAISS index with updated docs
        embeddings = OpenAIEmbeddings()
        faiss_index = FAISS.from_texts(docs, embeddings)
        
        # Store the new FAISS index
        st.session_state.uploaded_files = faiss_index
        st.session_state.uploaded_file_count = len(uploaded_files)
    st.success(f"Successfully indexed {len(docs)} document chunks.")


st.sidebar.header("⚙️ Model Settings")
st.session_state.model_choice = st.sidebar.selectbox("Choose Model", ["gpt-3.5-turbo", "gpt-4"], index=0)
st.session_state.model_creativity = st.sidebar.slider("Model Creativity (Temperature)", 0.0, 1.0, 0.7, 0.1)
st.session_state.response_length_words = st.sidebar.slider("Response Length (Words)", 50, 500, 150, 10)

if st.sidebar.button("Confirm Model Settings"):
    st.session_state.model_confirmed = True
    st.success("Model settings confirmed.")

# Displaying conversation history
for message in st.session_state.memory.chat_memory.messages:
    if isinstance(message, HumanMessage):
        st.chat_message("user").markdown(message.content)
    elif isinstance(message, AIMessage):
        st.chat_message("assistant").markdown(message.content)

if st.session_state.model_confirmed:
    
    query = st.chat_input("Ask a question:")

    if query:
        # Store user message in memory (this ensures the memory keeps track of user queries)
        st.session_state.memory.chat_memory.add_user_message(query)
        
        # Append the user message to the conversation history (for display)
        st.session_state.conversation_history.append({"role": "user", "content": query})
        
        if is_input_safe(query):
            if st.session_state.uploaded_files:
                # If the user uploaded files, use the retriever to get results
                retriever = st.session_state.uploaded_files.as_retriever(search_kwargs={"k": 2})
                qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", memory=st.session_state.memory)
                response = qa_chain.run(query)  # This returns a string
                
                # Only add response to memory and conversation history if it's a string
                if isinstance(response, str):
                    st.session_state.memory.chat_memory.add_ai_message(response)
                    st.session_state.conversation_history.append({"role": "assistant", "content": response})
                    st.chat_message("assistant").write(response)
            else:
                # If no file is uploaded, use the LLM directly
                system_message = SystemMessage(content=SYSTEM_PROMPT)
                user_message = HumanMessage(content=query)
                messages = st.session_state.memory.chat_memory.messages
                response = llm(messages + [system_message, user_message], 
                               temperature=st.session_state.model_creativity, 
                               max_tokens=512)  # response is an AIMessage object

                # Handle the assistant's response
                if isinstance(response, str):
                    st.session_state.memory.chat_memory.add_ai_message(response)
                    st.session_state.conversation_history.append({"role": "assistant", "content": response})
                    st.chat_message("assistant").write(response)
                else:
                    st.session_state.memory.chat_memory.add_ai_message(response.content)
                    st.session_state.conversation_history.append({"role": "assistant", "content": response.content})
                    st.chat_message("assistant").write(response.content)

        else:
            response = "⚠️ Your query violates content policies."
            st.session_state.memory.chat_memory.add_ai_message(response)
            st.session_state.conversation_history.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

else:
    st.warning("Confirm model settings before asking questions.")



def save_conversation_csv():
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Role", "Message"])

    # Extract messages from memory
    for msg in st.session_state.memory.chat_memory.messages:
        if isinstance(msg, HumanMessage):
            writer.writerow(["User", msg.content])
        elif isinstance(msg, AIMessage):
            writer.writerow(["Assistant", msg.content])

    return output.getvalue()

st.sidebar.header("💾 Download Conversation")
st.sidebar.download_button("Download CSV", save_conversation_csv(), "conversation.csv", "text/csv")
