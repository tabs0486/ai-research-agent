import os
from datetime import date

# Turn off CrewAI telemetry (must be set BEFORE importing crewai)
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from ddgs import DDGS

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="wide")


# ---------- 1. SEARCH TOOL (DuckDuckGo, free, no API key) ----------
@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """Search the web using DuckDuckGo. Input must be a plain search query string.
    Returns a numbered list of results with title, URL and a short snippet."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=6))
    except Exception as e:
        return f"Search failed: {e}. Try a different or simpler query."

    if not results:
        return "No results found. Try rephrasing the query."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"[{i}] {r.get('title', '')}\n"
            f"URL: {r.get('href', '')}\n"
            f"Snippet: {r.get('body', '')[:400]}"
        )
    return "\n\n".join(lines)


# ---------- 2. LLM (Groq via its OpenAI-compatible API) ----------
def get_llm() -> LLM:
    return LLM(
        # Groq's model id is "openai/gpt-oss-120b". CrewAI strips ONE leading
        # "openai/" prefix, so we add one extra to keep the real id intact.
        model="openai/openai/gpt-oss-120b",
        custom_openai=True,
        base_url="https://api.groq.com/openai/v1",
        api_key=st.secrets["GROQ_API_KEY"],
        temperature=0.2,
    )


# ---------- 3. THE SINGLE AGENT + TASK ----------
def run_research(topic: str) -> str:
    llm = get_llm()

    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Research the given topic thoroughly and write an accurate, well-organized report.",
        backstory=(
            "You are a careful analyst who searches the web, cross-checks several "
            "sources, and never invents facts or links."
        ),
        tools=[duckduckgo_search],
        llm=llm,
        allow_delegation=False,
        max_iter=8,       # stops the agent from looping forever
        verbose=False,
    )

    task = Task(
        description=(
            f"Research this topic: '{topic}'.\n"
            f"Today's date is {date.today().isoformat()}.\n"
            "Run several different searches to cover the basics, key facts, "
            "recent developments, and different viewpoints. "
            "Only use information found in the search results."
        ),
        expected_output=(
            "A Markdown report with: a title, an Executive Summary, "
            "Key Findings (bullet points), Recent Developments, "
            "Challenges or Different Viewpoints, a Conclusion, "
            "and a Sources section listing only URLs that appeared in search results."
        ),
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff()
    return result.raw


# ---------- 4. STREAMLIT UI ----------
st.title("🔎 AI Research Agent")
st.caption("Powered by CrewAI + Groq (gpt-oss-120b) + DuckDuckGo")

if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY is missing. Add it in Streamlit Cloud → App settings → Secrets.")
    st.stop()

topic = st.text_input("Enter a research topic", placeholder="e.g. Solid-state batteries")

if st.button("Start Research", type="primary"):
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        with st.spinner("Agent is searching and writing your report... (can take 1-2 minutes)"):
            try:
                report = run_research(topic.strip())
                st.session_state["report"] = report
            except Exception as e:
                st.error(f"Something went wrong: {e}")

if "report" in st.session_state:
    st.markdown("---")
    st.markdown(st.session_state["report"])
    st.download_button(
        "Download report (.md)",
        data=st.session_state["report"],
        file_name="research_report.md",
        mime="text/markdown",
    )
