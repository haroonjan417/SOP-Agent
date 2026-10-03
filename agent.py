import os
from crewai import Agent, Task, Crew, Process, LLM
from tools import (
    sop_search_rag, 
    web_search_tool, 
    generate_action_checklist, 
    draft_operational_artifacts
)

# --- FIX FOR GROQ CACHE BREAKPOINT ERROR ---
import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg
# -------------------------------------------

os.environ["PYTHONIOENCODING"] = "utf-8"

def sanitize_text(text: str) -> str:
    """Removes non-ASCII characters to prevent encoding issues."""
    return text.encode("ascii", "ignore").decode("ascii")

def run_sop_crew(user_incident: str, api_key: str, model_name: str = "groq/openai/gpt-oss-120b"):
    clean_incident = sanitize_text(user_incident)
    
    llm = LLM(
        model=model_name,
        api_key=api_key
    )
    
    sop_agent = Agent(
        role="Operations & Facilities Incident Manager",
        goal="Resolve operational incidents using internal SOPs if available, or fall back to web search and general industry best practices.",
        backstory=(
            "An expert operational manager specializing in logistics, facility maintenance, compliance, and automated risk response. "
            "You prioritize internal company SOPs. If internal SOPs do not cover an incident, you search the web "
            "and apply standard industrial/safety best practices to provide immediate, actionable guidance."
        ),
        tools=[sop_search_rag, web_search_tool, generate_action_checklist, draft_operational_artifacts],
        llm=llm,
        verbose=True
    )
    
    sop_task = Task(
        description=f"""
The following operational incident was reported:
'{clean_incident}'

Follow this decision hierarchy:
1. Search internal SOPs using 'SOP Policy Search Tool'.
2. IF internal SOP guidance IS FOUND:
   - Explicitly cite the document name (e.g., '[Document Source: Warehouse_SOP.pdf]') and section.
3. IF internal SOP guidance IS NOT FOUND or insufficient:
   - Explicitly note: "No specific internal SOP found for this issue. Falling back to general industry standards and web search."
   - Use 'Web Search Fallback Tool' to retrieve safety and troubleshooting guidelines for the issue.
4. Convert the retrieved guidance into an operational checklist and draft communications using the available tools.
5. Consolidate everything into a clear executive response for human approval.
""",
        expected_output=(
            "A structured report containing: "
            "1) Source Identification (Internal SOP Name OR Industry Best Practice Fallback), "
            "2) Immediate Action & Safety Steps, "
            "3) Action Checklist, and "
            "4) Pre-filled Operational Drafts/Tickets."
        ),
        agent=sop_agent
    )
    
    crew = Crew(
        agents=[sop_agent],
        tasks=[sop_task],
        process=Process.sequential
    )
    
    return crew.kickoff()
