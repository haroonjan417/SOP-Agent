import os
import streamlit as st
from agent import run_sop_crew
from tools import build_vectorstore_from_files

st.set_page_config(page_title="AI Multi-SOP Operations Agent", page_icon="⚙️", layout="wide")

st.title("⚙️ AI SOP-to-Action Business Process Agent")
st.caption("Powered by CrewAI, Groq API, Dynamic Multi-Document RAG, and Streamlit")

# Track indexed documents state
if "indexed_files_count" not in st.session_state:
    st.session_state["indexed_files_count"] = 0
if "indexed_file_names" not in st.session_state:
    st.session_state["indexed_file_names"] = []

# Retrieve API key automatically from Streamlit Secrets
groq_api_key = st.secrets.get("GROQ_API_KEY", "")

with st.sidebar:
    st.header("Configuration")
    
    if not groq_api_key:
        groq_api_key = st.text_input("Groq API Key", type="password")
    else:
        st.success("Groq API Key loaded securely from secrets!")

    selected_model = st.selectbox(
        "Groq Model",
        [
            "groq/openai/gpt-oss-120b",
            "groq/openai/gpt-oss-20b"
        ]
    )
    
    st.markdown("---")
    st.header("Upload SOP Documents")
    
    # Enable multiple document uploads
    uploaded_sops = st.file_uploader(
        "Upload Company SOPs (PDF or TXT)", 
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_sops:
        st.caption(f"📁 Selected {len(uploaded_sops)} document(s)")
        if st.button("Index / Reload SOP Knowledge Base"):
            with st.spinner("Indexing all uploaded SOP documents into Vector DB..."):
                count = build_vectorstore_from_files(uploaded_sops)
                st.session_state["indexed_files_count"] = count
                st.session_state["indexed_file_names"] = [f.name for f in uploaded_sops]
                st.success(f"Successfully indexed {count} document(s)!")

    if st.session_state["indexed_files_count"] > 0:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Currently Active SOPs:**")
        for fname in st.session_state["indexed_file_names"]:
            st.sidebar.markdown(f"- `{fname}`")

st.markdown("### Report an Operational Incident")
st.info("Example: *'5 crates of raw material arrived damaged at Loading Bay 2. Packaging is ruptured.'*")

user_incident = st.text_area("Describe the operational issue or event:", height=100)

if st.button("Analyze & Generate Action Plan", type="primary"):
    # Auto-index on first run if user uploaded files but didn't explicitly click 'Index'
    if uploaded_sops and st.session_state["indexed_files_count"] == 0:
        with st.spinner("Indexing uploaded SOP documents..."):
            count = build_vectorstore_from_files(uploaded_sops)
            st.session_state["indexed_files_count"] = count
            st.session_state["indexed_file_names"] = [f.name for f in uploaded_sops]

    if not groq_api_key:
        st.error("Missing Groq API key in secrets or sidebar.")
    elif not user_incident.strip():
        st.warning("Please provide an incident description.")
    elif st.session_state["indexed_files_count"] == 0:
        st.warning("Please upload and index at least one SOP document (PDF/TXT) in the sidebar first.")
    else:
        os.environ["GROQ_API_KEY"] = groq_api_key
        
        with st.spinner("Agent evaluating SOPs across uploaded documents..."):
            try:
                result = run_sop_crew(
                    user_incident=user_incident, 
                    api_key=groq_api_key, 
                    model_name=selected_model
                )
                st.success("Action Plan Ready for Human Approval")
                st.markdown("---")
                st.markdown(result.raw)
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
