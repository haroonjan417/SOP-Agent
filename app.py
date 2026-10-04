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

# Custom CSS for Modern Dark/Indigo Vibe
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
    
    /* Headers & Title Accent */
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-family: 'Inter', sans-serif;
    }
    
    .stTitle {
        background: linear-gradient(90deg, #6366F1 0%, #A855F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Cards / Container Boxes */
    div[data-testid="stExpander"], div.stTextArea {
        background-color: #1E293B;
        border-radius: 10px;
        border: 1px solid #334155;
    }

    /* Primary Action Buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    /* Team Card Styling */
    .team-card {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        margin-top: 10px;
    }
    .team-leader {
        color: #818CF8;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .team-member {
        color: #94A3B8;
        font-size: 0.9rem;
        margin-left: 4px;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.title("⚡ AI SOP Operations Agent")
st.caption("Powered by CrewAI, Groq API, Dynamic Multi-Doc RAG & Streamlit")

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

    st.markdown("---")
    
    # --- TEAM ATTRIBUTION SECTION ---
    st.markdown("### 👥 Project Team")
    st.markdown("""
    <div class="team-card">
        <div class="team-leader">👑 Team Leader:</div>
        <div style="color: #F8FAFC; font-weight: 600; margin-left: 4px;">Masood ur Rahman</div>
        <hr style="border-color: #334155; margin: 8px 0;">
        <div class="team-leader">🤝 Team Members:</div>
        <div class="team-member">• M Haroon Jan</div>
        <div class="team-member">• Fatima Ijaz</div>
        <div class="team-member">• Aslam Afridi</div>
        <div class="team-member">• Shakeel Ahmad</div>
        <div class="team-member">• Sami Ur Rahman</div>
    </div>
    """, unsafe_allow_html=True)

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
