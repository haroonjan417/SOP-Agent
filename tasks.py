from crewai import Agent, Task, Crew, Process, LLM


def run_sop_agent_workflow(
    user_incident: str,
    sop_context: str,
    api_key: str,
    model_name: str = "groq/openai/gpt-oss-120b"
):
    """
    Single-agent SOP workflow.

    The workflow combines:
    1. RAG-retrieved SOP context
    2. Agentic reasoning
    3. Operational workflow generation
    4. Human approval
    """

    # ---------------------------------------------------------
    # 1. Create the LLM
    # ---------------------------------------------------------

    llm = LLM(
        model=model_name,
        api_key=api_key
    )

    # ---------------------------------------------------------
    # 2. Create ONE SOP Agent
    # ---------------------------------------------------------

    sop_agent = Agent(
        role="Senior SOP Operations Agent",

        goal=(
            "Analyze operational incidents using the retrieved company "
            "SOP information and create a safe, practical and actionable "
            "response for human approval."
        ),

        backstory=(
            "You are an experienced operations and compliance manager. "
            "You specialize in interpreting company SOPs, identifying "
            "operational risks, creating prioritized action plans, and "
            "drafting business communications. "
            "You must rely primarily on the provided SOP context and "
            "must never invent company policies."
        ),

        llm=llm,

        verbose=True,

        allow_delegation=False
    )

    # ---------------------------------------------------------
    # 3. Give the agent ONE complete workflow
    # ---------------------------------------------------------

    sop_task = Task(
        description=f"""
You are processing the following operational incident:

INCIDENT:
{user_incident}

The following information was retrieved from the company's
SOP knowledge base using RAG:

RETRIEVED SOP CONTEXT:
{sop_context}

Follow this workflow carefully.

STEP 1 — SOP ANALYSIS
Identify the SOP procedures that are relevant to the incident.

STEP 2 — SOURCE IDENTIFICATION
Clearly identify the SOP document source provided in the
retrieved context.

STEP 3 — RISK ANALYSIS
Explain the important operational, safety or compliance
risks associated with the incident.

STEP 4 — ACTION PLAN
Create a prioritized action checklist.

Use:
- HIGH priority
- MEDIUM priority
- LOW priority

STEP 5 — BUSINESS AUTOMATION
Prepare an appropriate operational communication or artifact,
such as:
- supplier communication
- maintenance ticket
- incident report
- NCR request
- internal notification

STEP 6 — HUMAN APPROVAL
The final recommendation must clearly state:

PENDING MANAGER APPROVAL

IMPORTANT RULES:

1. Use the retrieved SOP context as the primary source.
2. Do not invent SOP rules that are not supported by the context.
3. If the retrieved SOP information is insufficient, clearly say so.
4. Make the output practical for an operations manager.
5. Keep the response structured and easy to read.
""",

        expected_output=(
            "A professional markdown operational report containing: "
            "1. Incident Summary, "
            "2. SOP Source and Relevant Procedures, "
            "3. Risk Analysis, "
            "4. Prioritized Action Checklist, "
            "5. Operational Communication/Draft Artifact, "
            "6. PENDING MANAGER APPROVAL status."
        ),

        agent=sop_agent
    )

    # ---------------------------------------------------------
    # 4. Run ONE agent
    # ---------------------------------------------------------

    crew = Crew(
        agents=[sop_agent],
        tasks=[sop_task],
        process=Process.sequential,
        verbose=True
    )

    return crew.kickoff()
```
