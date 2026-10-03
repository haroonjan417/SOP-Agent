import os
import streamlit as st
from agent import run_sop_crew

st.set_page_config(page_title="AI SOP Operations Agent", page_icon="⚙️", layout="wide")

st.title("⚙️ AI SOP-to-Action Business Process Agent")
st.caption("Powered by CrewAI, Groq API (openai/gpt-oss-120b), RAG, and Streamlit")

with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Get your free key from console.groq.com")
    selected_model = st.text_input("Model ID", value="groq/openai/gpt-oss-120b")

st.markdown("### Report an Operational Incident")
st.info("Example: *'5 crates of raw material arrived damaged at Loading Bay 2. Packaging is ruptured.'*")

user_incident = st.text_area("Describe the operational issue or event:", height=100)

if st.button("Analyze & Generate Action Plan", type="primary"):
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    elif not user_incident.strip():
        st.warning("Please provide an incident description.")
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
