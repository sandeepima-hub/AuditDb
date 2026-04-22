import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
from io import BytesIO
from docx import Document

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
import tempfile

# --- 1. SECRETS & CONFIGURATION ---
# These will be pulled securely from Streamlit Cloud Secrets
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

index_name = "audit-db"

# --- 2. AUTHENTICATION SETUP ---
# Fetch credentials securely from Streamlit Secrets
credentials = dict(st.secrets["credentials"])
cookie = dict(st.secrets["cookie"])

authenticator = stauth.Authenticate(
    credentials,
    cookie["name"],
    cookie["key"],
    cookie["expiry_days"]
)

# --- 3. LOGIN UI ---
st.set_page_config(page_title="Cloud Audit AI", layout="centered")
authenticator.login("main")

if st.session_state["authentication_status"] is False:
    st.error("Username/password is incorrect")
elif st.session_state["authentication_status"] is None:
    st.warning("Please enter your username and password")
elif st.session_state["authentication_status"]:
    
    # --- 4. THE MAIN APPLICATION (Only visible if logged in) ---
    
    # Logout button in sidebar
    with st.sidebar:
        st.write(f'Welcome *{st.session_state["name"]}*')
        authenticator.logout("Logout", "sidebar")
        st.divider()

    st.title("🛡️ Secure Cloud Audit Engine")
    
    # Initialize Cloud AI Models
    # Using Llama-3-8b via Groq for high-speed reasoning
    llm = ChatGroq(model_name="llama3-8b-8192")
    # Using a fast, free embedding model via HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    # Sidebar: Document Upload
    with st.sidebar:
        st.header("Document Repository")
        uploaded_file = st.file_uploader("Upload Audit PDF", type=["pdf"])
        
        if st.button("Process & Upload to Cloud"):
            if uploaded_file is not None:
                with st.spinner("Processing..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    try:
                        loader = PyPDFLoader(tmp_path)
                        docs = loader.load()
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                        chunks = text_splitter.split_documents(docs)
                        
                        for chunk in chunks:
                            chunk.metadata["source"] = uploaded_file.name
                        
                        vector_store.add_documents(chunks)
                        st.success(f"Uploaded {uploaded_file.name} to Pinecone!")
                    finally:
                        os.remove(tmp_path)
            else:
                st.warning("Please upload a file.")

    # Main UI: Generate Questionnaire
    focus_area = st.text_input("Audit Focus Area", placeholder="e.g., Data Privacy Protocol")
    
    # Hardcoded document name input for simplicity in this MVP (can be fetched dynamically via Pinecone API)
    doc_target = st.text_input("Target Document Name (Exact PDF name uploaded)", placeholder="e.g., policy.pdf")

    if st.button("Generate Questionnaire"):
        if focus_area and doc_target:
            with st.spinner("Querying Cloud AI..."):
                retriever = vector_store.as_retriever(
                    search_kwargs={"k": 4, "filter": {"source": doc_target}}
                )
                retrieved_docs = retriever.invoke(focus_area)
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                prompt = f"""You are an expert auditor. Based ONLY on the excerpts below from {doc_target}, generate a 5-question audit checklist about: {focus_area}.
                Excerpts: {context}"""
                
                response = llm.invoke(prompt)
                
                st.session_state['last_result'] = response.content
                st.session_state['last_focus'] = focus_area
                st.markdown("### Generated Questionnaire")
                st.write(response.content)

    # Word Export
    if 'last_result' in st.session_state:
        st.divider()
        st.subheader("Export")
        
        doc = Document()
        doc.add_heading(f"Audit Questionnaire: {st.session_state.get('last_focus', 'Topic')}", 0)
        doc.add_paragraph(st.session_state['last_result'])
        
        bio = BytesIO()
        doc.save(bio)
        
        st.download_button(
            label="Download as Word (.docx)",
            data=bio.getvalue(),
            file_name="audit_questionnaire.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )