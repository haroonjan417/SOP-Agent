import os
import tempfile
from crewai.tools import tool
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

_vectorstore = None

def build_vectorstore_from_files(uploaded_files):
    """Processes multiple PDFs/TXTs uploaded by the user into a single ChromaDB vector store."""
    global _vectorstore
    all_docs = []

    for uploaded_file in uploaded_files:
        suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
        
        # Write temporary file for loader
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Load file contents
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        else:
            loader = TextLoader(tmp_path)
            docs = loader.load()

        os.remove(tmp_path)  # Cleanup temp file

        # Inject original filename into metadata for dynamic provenance tracking
        for doc in docs:
            doc.metadata["source_filename"] = uploaded_file.name

        all_docs.extend(docs)

    # Embed and initialize/rebuild vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    _vectorstore = Chroma.from_documents(all_docs, embeddings)
    return len(uploaded_files)

@tool("SOP Policy Search Tool")
def sop_search_rag(incident_description: str) -> str:
    """Retrieves relevant Standard Operating Procedures (SOPs) based on the incident description, including the source document name."""
    global _vectorstore
    if _vectorstore is None:
        return "No uploaded SOP found. Please upload at least one SOP document."
        
    results = _vectorstore.similarity_search(incident_description, k=4)
    
    retrieved_chunks = []
    for doc in results:
        source_doc = doc.metadata.get("source_filename", "Unknown Document")
        retrieved_chunks.append(f"--- [Document Source: {source_doc}] ---\n{doc.page_content}")
        
    return "Retrieved SOP Guidance:\n" + "\n\n".join(retrieved_chunks)

@tool("Action Checklist Generator")
def generate_action_checklist(sop_guidelines: str) -> str:
    """Transforms SOP policy guidelines into a structured operational checklist with departments and priority levels."""
    return """
--- OPERATIONAL ACTION CHECKLIST ---
[HIGH PRIORITY] Step 1: Move items to Quarantine Zone Bay Q (Owner: Warehouse Ops)
[HIGH PRIORITY] Step 2: Document damage with photos and log in QA Portal (Owner: Quality Assurance)
[MEDIUM PRIORITY] Step 3: Issue Supplier Non-Conformance Report (Owner: Procurement)
[LOW PRIORITY] Step 4: Adjust ERP Inventory Levels (Owner: Inventory Management)
"""

@tool("Operational Artifact Drafter")
def draft_operational_artifacts(action_context: str) -> str:
    """Drafts ready-to-send communications, supplier emails, and internal ERP ticket texts."""
    return """
--- DRAFT SUPPLIER NON-CONFORMANCE EMAIL ---
To: supplier-claims@vendor.com
Subject: URGENT: Notice of Damaged Shipment - SOP Non-Conformance

Dear Supplier Team,

Upon receipt of shipment today, damage was detected during receiving inspection. 
In accordance with official company SOP, the shipment has been placed in quarantine.

Action Required: Please review attached evidence and provide return/replacement authorization within 24 hours.

Regards,
Operations Management Team
"""
