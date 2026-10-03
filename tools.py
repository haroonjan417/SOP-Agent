import os
from crewai.tools import tool
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "sops")
_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        loader = DirectoryLoader(DATA_DIR, glob="*.txt", loader_cls=TextLoader)
        docs = loader.load()
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _vectorstore = Chroma.from_documents(docs, embeddings)
    return _vectorstore

@tool("SOP Policy Search Tool")
def sop_search_rag(incident_description: str) -> str:
    """Retrieves relevant Standard Operating Procedures (SOPs) based on the incident description."""
    vs = get_vectorstore()
    results = vs.similarity_search(incident_description, k=2)
    policies = "\n\n".join([doc.page_content for doc in results])
    return f"Retrieved SOP Guidance:\n{policies}"

@tool("Action Checklist Generator")
def generate_action_checklist(sop_guidelines: str) -> str:
    """Transforms SOP policy guidelines into a structured operational checklist with departments and priority levels."""
    return f"""
    --- OPERATIONAL ACTION CHECKLIST ---
    [HIGH PRIORITY] Step 1: Move items to Quarantine Zone Bay Q (Owner: Warehouse Ops)
    [HIGH PRIORITY] Step 2: Document damage with photos & log in QA Portal (Owner: Quality Assurance)
    [MEDIUM PRIORITY] Step 3: Issue Supplier Non-Conformance Report (Owner: Procurement)
    [LOW PRIORITY] Step 4: Adjust ERP Inventory Levels (Owner: Inventory Management)
    """

@tool("Operational Artifact Drafter")
def draft_operational_artifacts(action_context: str) -> str:
    """Drafts ready-to-send communications, supplier emails, and internal ERP ticket texts."""
    return f"""
    --- DRAFT SUPPLIER NON-CONFORMANCE EMAIL ---
    To: supplier-claims@vendor.com
    Subject: URGENT: Notice of Damaged Shipment - SOP Non-Conformance

    Dear Supplier Team,

    Upon receipt of shipment today, damage was detected during receiving inspection. 
    In accordance with SOP-804, the shipment has been placed in quarantine.

    Action Required: Please review attached evidence and provide return/replacement authorization within 24 hours.

    Regards,
    Operations Management Team
    """
