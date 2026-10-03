import os
from crewai import Agent, Task, Crew, Process, LLM
from tools import sop_search_rag, generate_action_checklist, draft_operational_artifacts

# Ensure environment uses UTF-8 encoding
os.environ["PYTHONIOENCODING"] = "utf-8"

def sanitize_text(text: str) -> str:
    """Removes non-ASCII characters to prevent LiteLLM/Groq encoding issues."""
    return text.encode("ascii", "ignore").decode("ascii")

def run_sop_crew(user_incident: str, api_key: str, model_name: str = "groq/llama-3.3-70b-versatile"):
    # Clean non-ASCII characters from input prompt
    clean_incident = sanitize_text(user_incident)
    
    llm = LLM(
        model=model_name,
        api_key=api_key
    )
    
    sop_agent = Agent(
        role="Operations SOP and Execution Manager",
        goal="Analyze incidents against company SOPs using tools and generate actionable resolution workflows.",
        backstory="An expert operational manager specializing in logistics, compliance, and automated risk response.",
        tools=[sop_search_rag, generate_action_checklist, draft_operational_artifacts],
        llm=llm,
        verbose=True
    )
    
    task = Task(
        description=f"""
The following operational incident was reported:
'{clean_incident}'

Follow this exact sequence:
1. Use the 'SOP Policy Search Tool' to retrieve relevant procedures for this incident.
2. Use the 'Action Checklist Generator' to turn retrieved guidelines into a structured task list.
3. Use the 'Operational Artifact Drafter' to prepare draft communications or tickets.
4. Consolidate everything into a clear executive response for human approval.
""",
        expected_output="A structured report containing: 1) Applicable SOP Summary, 2) Departmental Action Checklist, and 3) Pre-filled Operational Drafts.",
        agent=sop_agent
    )
    
    crew = Crew(
        agents=[sop_agent],
        tasks=[task],
        process=Process.sequential
    )
    
    return crew.kickoff()
