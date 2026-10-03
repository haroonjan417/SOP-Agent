import os
import streamlit as st
from agent import run_sop_crew
from tools import build_vectorstore_from_file

st.set_page_config(page_title="AI SOP Operations Agent", page_icon="⚙️", layout="wide")

st.title("⚙️ AI SOP-to-Action Business Process Agent")
st.caption("Powered by CrewAI, Groq API, Dynamic RAG, and Streamlit")

# Track indexed state in session
if "sop_indexed" not in st.session_state:
    st.session_state["sop_indexed"] = False

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
    st.header("Upload SOP Document")
    uploaded_sop = st.file_uploader("Upload Company SOP (PDF or TXT)", type=["pdf", "txt"])

    # Automatically index when a new file is uploaded
    if uploaded_sop is not None:
        if st.sidebar.button("Index / Reload SOP"):
            with st.spinner("Indexing uploaded SOP document..."):
                build_vectorstore_from_file(uploaded_sop)
                st.session_state["sop_indexed"] = True
                st.sidebar.success(f"Indexed: {uploaded_sop.name}")

st.markdown("### Report an Operational Incident")
st.info("Example: *'5 crates of raw material arrived damaged at Loading Bay 2. Packaging is ruptured.'*")

user_incident = st.text_area("Describe the operational issue or event:", height=100)

if st.button("Analyze & Generate Action Plan", type="primary"):
    # If a file is in the uploader but hasn't been indexed yet, index it automatically
    if uploaded_sop is not None and not st.session_state["sop_indexed"]:
        with st.spinner("Indexing uploaded SOP document..."):
            build_vectorstore_from_file(uploaded_sop)
            st.session_state["sop_indexed"] = True

    if not groq_api_key:
        st.error("Missing Groq API key in secrets or sidebar.")
    elif not user_incident.strip():
        st.warning("Please provide an incident description.")
    elif not st.session_state["sop_indexed"]:
        st.warning("Please upload an SOP document (PDF/TXT) in the sidebar first.")
    else:
        os.environ["GROQ_API_KEY"] = groq_api_key
        
        with st.spinner("Agent evaluating SOPs and generating workflow..."):
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
