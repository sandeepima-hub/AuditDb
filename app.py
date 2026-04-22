import streamlit as st
import os
import time
import requests
import json
import tempfile
import pandas as pd
from datetime import datetime
from io import BytesIO

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.embeddings import Embeddings

# --- 1. SECRETS & CONFIGURATION ---
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
index_name = "audit-db"

# --- 2. PERSISTENCE (STATE MANAGEMENT) ---
STATE_FILE = "audit_state.json"

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"documents": [], "history": []}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

if "app_state" not in st.session_state:
    st.session_state.app_state = load_state()

# --- 3. CUSTOM BULLETPROOF EMBEDDER ---
class SafeHFEmbeddings(Embeddings):
    def __init__(self, api_key, model):
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def embed_documents(self, texts):
        for attempt in range(3):
            payload = {"inputs": texts, "options": {"wait_for_model": True}}
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            if response.status_code == 200:
                try:
                    return response.json()
                except Exception:
                    raise Exception(f"API returned invalid data. Raw response: {response.text}")
            elif response.status_code == 503:
                time.sleep(5)
                continue
            else:
                raise Exception(f"HF Error {response.status_code}: {response.text}")
        raise Exception("Hugging Face API timed out. The server is currently too busy.")

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# --- 4. INITIALIZE AI MODELS ---
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.2) # Lower temp for structured data
embeddings = SafeHFEmbeddings(api_key=hf_token, model="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# --- 5. UI SETUP & SIDEBAR (MULTI-DOC UPLOAD) ---
st.set_page_config(page_title="Enterprise Audit AI", layout="wide")
st.title("🛡️ Enterprise Audit Engine")

with st.sidebar:
    st.header("📂 Document Repository")
    # Feature 1: Multi-Document Upload
    uploaded_files = st.file_uploader("Upload Audit PDFs", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process & Upload to Cloud"):
        if uploaded_files:
            with st.spinner(f"Processing {len(uploaded_files)} files..."):
                for uploaded_file in uploaded_files:
                    # Skip if already processed
                    if uploaded_file.name in st.session_state.app_state["documents"]:
                        st.info(f"Skipping {uploaded_file.name} (already in database).")
                        continue
                        
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    try:
                        loader = PyPDFLoader(tmp_path)
                        docs = loader.load()
                        
                        # Feature 2: Improved Chunking (Respects paragraphs and sentences)
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=800,
                            chunk_overlap=150,
                            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
                        )
                        chunks = text_splitter.split_documents(docs)
                        
                        # Feature 3: Rich Metadata Injection
                        current_date = datetime.now().strftime("%Y-%m-%d")
                        for i, chunk in enumerate(chunks):
                            chunk.metadata["source"] = uploaded_file.name
                            chunk.metadata["chunk_id"] = f"{uploaded_file.name}_chunk_{i}"
                            chunk.metadata["upload_date"] = current_date
                            # PyPDFLoader automatically adds "page" metadata, we ensure it exists
                            chunk.metadata["page"] = chunk.metadata.get("page", "Unknown") 
                        
                        # Batch Upload
                        batch_size = 10
                        for i in range(0, len(chunks), batch_size):
                            vector_store.add_documents(chunks[i : i + batch_size])
                            time.sleep(1)
                            
                        # Update Persistence State
                        st.session_state.app_state["documents"].append(uploaded_file.name)
                        save_state(st.session_state.app_state)
                        st.success(f"Successfully integrated {uploaded_file.name}!")
                        
                    except Exception as e:
                        st.error(f"Failed to process {uploaded_file.name}: {str(e)}")
                    finally:
                        os.remove(tmp_path)
        else:
            st.warning("Please upload at least one file.")

    st.divider()
    st.markdown("### Available Documents")
    for doc in st.session_state.app_state["documents"]:
        st.markdown(f"📄 `{doc}`")

# --- 6. MAIN TABS (GENERATE & HISTORY) ---
tab1, tab2 = st.tabs(["🔍 Generate Audit", "📚 Previous Audits"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        focus_area = st.text_input("Audit Focus Area", placeholder="e.g., Data Privacy Protocol")
    with col2:
        # Feature 1b: Dropdown Selection powered by persisted state
        doc_options = ["All Documents"] + st.session_state.app_state["documents"]
        doc_target = st.selectbox("Target Document", options=doc_options)

    if st.button("Generate Structured Audit", type="primary"):
        if focus_area and st.session_state.app_state["documents"]:
            with st.spinner("Extracting parameters and structuring tables..."):
                try:
                    # Filter logic based on dropdown
                    search_kwargs = {"k": 6}
                    if doc_target != "All Documents":
                        search_kwargs["filter"] = {"source": doc_target}
                        
                    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
                    retrieved_docs = retriever.invoke(focus_area)
                    
                    if not retrieved_docs:
                        st.warning("No relevant text found.")
                    else:
                        # Build context including our new metadata
                        context = ""
                        for doc in retrieved_docs:
                            pg = doc.metadata.get('page', 'N/A')
                            src = doc.metadata.get('source', 'Unknown')
                            context += f"\n--- [Source: {src} | Page: {pg}] ---\n{doc.page_content}\n"
                        
                        # Feature 4: Prompt Engineering for Structured JSON Output
                        prompt = f"""You are a strict, expert auditor. Based ONLY on the excerpts provided, generate a 5-question audit checklist about: {focus_area}.
                        
                        You MUST respond with ONLY a raw JSON array. Do not include markdown formatting, backticks, or conversational text. 
                        The JSON must follow this exact schema:
                        [
                            {{
                                "Checklist Item": "The specific question to ask",
                                "Rationale": "Why this matters based on the text",
                                "Source Paragraph": "The exact quote from the text that justifies this",
                                "Document & Page": "The source file and page number provided in the context"
                            }}
                        ]
                        
                        Excerpts: 
                        {context}"""
                        
                        response = llm.invoke(prompt)
                        
                        # Clean the LLM output to ensure it's pure JSON
                        clean_json = response.content.replace("```json", "").replace("```", "").strip()
                        audit_data = json.loads(clean_json)
                        
                        # Convert to Pandas DataFrame for a beautiful table
                        df = pd.DataFrame(audit_data)
                        
                        # Save to history (Feature 5)
                        audit_record = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "focus": focus_area,
                            "target": doc_target,
                            "data": audit_data
                        }
                        st.session_state.app_state["history"].insert(0, audit_record) # Add to top
                        save_state(st.session_state.app_state)
                        
                        st.markdown("### Structured Audit Output")
                        st.dataframe(df, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error generating audit: {str(e)}")
        else:
            st.warning("Please define a focus area and ensure documents are uploaded.")

# Feature 5: History Tab
with tab2:
    st.markdown("### Previously Generated Audits")
    if not st.session_state.app_state["history"]:
        st.info("No audits generated yet.")
    else:
        for idx, record in enumerate(st.session_state.app_state["history"]):
            with st.expander(f"Audit: {record['focus']} ({record['timestamp']})"):
                st.markdown(f"**Target:** `{record['target']}`")
                df_history = pd.DataFrame(record["data"])
                st.dataframe(df_history, use_container_width=True)
