import os
import streamlit as st
from agent import run_sop_crew
from tools import build_vectorstore_from_files

# --- MODERN UI PAGE CONFIG & STYLING ---
st.set_page_config(
    page_title="AI SOP Operations Agent", 
    page_icon="⚡", 
    layout="wide"
)

# Custom CSS for Modern Dark/Light Orange Vibe
st.markdown("""
<style>
    /* Main Background & Text Color */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }
    
    /* Headers & Title Accent - Light Orange / Amber Gradient */
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-family: 'Inter', sans-serif;
    }
    
    .stTitle {
        background: linear-gradient(90deg, #FF8C00 0%, #FFA500 50%, #FFC107 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0px;
    }
    
    /* Main Banner Team Card Styling with Orange Highlights */
    .hero-team-card {
        background-color: #1E293B;
        border: 1px solid #FF8C00;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 10px;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(255, 140, 0, 0.1);
    }
    .hero-leader {
        color: #FF9800;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .hero-leader span {
        color: #F8FAFC;
        font-weight: 600;
    }
    .hero-members {
        color: #94A3B8;
        font-size: 0.9rem;
    }
    .hero-members span {
        color: #CBD5E1;
        font-weight: 500;
    }

    /* Cards & Input Area Styling */
    div[data-testid="stExpander"], div.stTextArea {
        background-color: #1E293B;
        border-radius: 10px;
        border: 1px solid #334155;
    }

    /* Force Input Text to be Visible (White Text on Dark Field) */
    div[data-baseweb="textarea"] textarea, div[data-baseweb="input"] input {
        color: #FFFFFF !important;
        background-color: #1E293B !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    /* Input Focus Outline - Vibrant Orange */
    div[data-baseweb="textarea"]:focus-within, div[data-baseweb="input"]:focus-within {
        border-color: #FF8C00 !important;
    }

    /* Placeholder Text Visibility */
    div[data-baseweb="textarea"] textarea::placeholder {
        color: #94A3B8 !important;
        -webkit-text-fill-color: #94A3B8 !important;
    }

    /* Primary Action Buttons - Light Orange to Amber Gradient */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #FF8C00 0%, #FF7A00 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #FF9800 0%, #FF8C00 100%);
        opacity: 0.95;
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(255, 140, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.title("⚡ AI SOP Operations Agent")
st.caption("Powered by CrewAI, Groq API, Dynamic Multi-Doc RAG & Streamlit")

# --- TEAM ATTRIBUTION BANNER (UNDER TITLE) ---
st.markdown("""
<div class="hero-team-card">
    <div class="hero-leader">👑 Team Leader: <span>Masood ur Rahman</span></div>
    <div class="hero-members">🤝 Team Members: <span>M Haroon Jan &nbsp;•&nbsp; Fatima Ijaz &nbsp;•&nbsp; Aslam Afridi &nbsp;•&nbsp; Shakeel Ahmad &nbsp;•&nbsp; Sami Ur Rahman</span></div>
</div>
""", unsafe_allow_html=True)

# --- TRACK SESSION STATE ---
if "indexed_files_count" not in st.session_state:
    st.session_state["indexed_files_count"] = 0
if "indexed_file_names" not in st.session_state:
    st.session_state["indexed_file_names"] = []

# Retrieve API key automatically from Streamlit Secrets
groq_api_key = st.secrets.get("GROQ_API_KEY", "").strip()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    if not groq_api_key:
        groq_api_key = st.text_input("Groq API Key", type="password")
    else:
        st.success("Groq API Key loaded securely")

    selected_model = st.selectbox(
        "Groq Model",
        [
            "groq/openai/gpt-oss-120b",
            "groq/llama-3.3-70b-versatile",
            "groq/llama3-70b-8192"
        ]
    )
    
    st.markdown("---")
    st.header("📁 SOP Knowledge Base")
    
    uploaded_sops = st.file_uploader(
        "Upload Company SOPs (PDF or TXT)", 
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_sops:
        st.caption(f"Selected {len(uploaded_sops)} document(s)")
        if st.button("Index / Reload SOP Knowledge Base"):
            with st.spinner("Indexing uploaded SOP documents..."):
                count = build_vectorstore_from_files(uploaded_sops)
                st.session_state["indexed_files_count"] = count
                st.session_state["indexed_file_names"] = [f.name for f in uploaded_sops]
                st.success(f"Successfully indexed {count} document(s)!")

    if st.session_state["indexed_files_count"] > 0:
        st.markdown("**Active SOPs:**")
        for fname in st.session_state["indexed_file_names"]:
            st.markdown(f"- `{fname}`")

# --- MAIN APP INPUT SECTION ---
st.markdown("### 🚨 Report an Operational Incident")
st.info("Example: *'5 crates of raw material arrived damaged at Loading Bay 2. Packaging is ruptured.'*")

user_incident = st.text_area("Describe the operational issue or event:", height=120)

if st.button("Analyze & Generate Action Plan", type="primary"):
    if uploaded_sops and st.session_state["indexed_files_count"] == 0:
        with st.spinner("Indexing uploaded SOP documents..."):
            count = build_vectorstore_from_files(uploaded_sops)
            st.session_state["indexed_files_count"] = count
            st.session_state["indexed_file_names"] = [f.name for f in uploaded_sops]

    if not groq_api_key:
        st.error("Missing Groq API key in secrets or sidebar.")
    elif not user_incident.strip():
        st.warning("Please provide an incident description.")
    else:
        os.environ["GROQ_API_KEY"] = groq_api_key
        
        with st.spinner("Agent evaluating incident and generating workflow..."):
            try:
                result = run_sop_crew(
                    user_incident=user_incident, 
                    api_key=groq_api_key, 
                    model_name=selected_model
                )
                st.success("Action Plan Ready for Approval")
                st.markdown("---")
                st.markdown(result.raw)
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
