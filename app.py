import os
import litellm
import streamlit as st
from crewai import Agent, Task, Crew, LLM
from tools import build_vectorstore_from_files, search_sop_database

# Clean and validate the key input
clean_api_key = groq_api_key.strip().replace('"', '').replace("'", "")

if not clean_api_key.startswith("gsk_"):
    st.error("Invalid key format. Groq API keys must start with 'gsk_'.")
    st.stop()

# Set environment variable so LiteLLM reads it directly
os.environ["GROQ_API_KEY"] = clean_api_key

# Prevent LiteLLM from sending unsupported parameters (like cache_breakpoint) to Groq
litellm.drop_params = True

st.set_page_config(page_title="SOP Assistant", layout="wide")
st.title("📄 Single-Agent SOP Assistant")

# Sidebar - Indexing
with st.sidebar:
    st.header("SOP Document Indexer")
    groq_api_key = st.text_input("Groq API Key", type="password")
    uploaded_sops = st.file_uploader("Upload SOP Files", type=["pdf", "txt"], accept_multiple_files=True)
    index_button = st.button("Index SOPs")

    if index_button:
        if not uploaded_sops:
            st.warning("Please upload at least one file.")
        else:
            with st.spinner("Processing and indexing SOP documents..."):
                count = build_vectorstore_from_files(uploaded_sops)
                if count > 0:
                    st.success(f"Successfully indexed {count} SOP document(s)!")
                else:
                    st.error("Failed to index uploaded documents.")

# Initialize chat session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Chat Input
user_query = st.chat_input("Ask a question about your indexed SOPs...")

if user_query:
    if not groq_api_key:
        st.error("Please provide a Groq API Key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing SOPs..."):
            try:
                # Configure Single LLM instance
                groq_llm = LLM(
                    model="openai/gpt-oss-120b",
                    api_key=clean_api_key,
                    drop_params=True
                )

                # Single Unified SOP Agent
                sop_agent = Agent(
                    role="Senior SOP Specialist",
                    goal="Accurately extract, summarize, and answer questions based on indexed SOP documents.",
                    backstory="You are an expert technical auditor specializing in Standard Operating Procedures. "
                              "You retrieve information directly from the SOP database using your tool and give concise, clear responses.",
                    tools=[search_sop_database],
                    llm=groq_llm,
                    verbose=True
                )

                # Single Task
                sop_task = Task(
                    description=f"Answer the user query using the SOP Vector Search Tool: '{user_query}'",
                    expected_output="A precise, structured answer referencing specific SOP facts and source files where applicable.",
                    agent=sop_agent
                )

                # Execute Single Agent Crew
                crew = Crew(
                    agents=[sop_agent],
                    tasks=[sop_task]
                )

                result = crew.kickoff()
                response_text = str(result)

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
