import time
import streamlit as st

from src.agents.agents import (
    build_read_agent,
    build_search_agent,
    write_chain,
    critic_chain,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_text(content) -> str:
    """Normalize LLM message content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


@st.cache_resource(show_spinner=False)
def get_agents():
    """Build agents once and reuse them across Streamlit reruns."""
    return build_search_agent(), build_read_agent()


# ---------------------------------------------------------------------------
# Pipeline (UI-independent logic)
# ---------------------------------------------------------------------------

def run_research_pipeline(topic: str, on_step=None):
    """
    Runs the full multi-agent pipeline.

    on_step(step_name, status, output=None) is an optional callback used by
    the UI to show progress. status is one of: "start", "done".
    """

    def notify(step, status, output=None):
        if on_step:
            on_step(step, status, output)

    search_agent, read_agent = get_agents()
    timings = {}

    # Step 1: Search the web
    notify("search", "start")
    t0 = time.time()
    search_result = search_agent.invoke({
        "messages": [{"role": "user", "content": topic}]
    })
    search_output = _to_text(search_result["messages"][-1].content)
    timings["search"] = time.time() - t0
    notify("search", "done", search_output)

    # Step 2: Read and analyze the search findings
    notify("read", "start")
    t0 = time.time()
    reader_result = read_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Research topic: {topic}\n\n"
                    f"Search findings:\n{search_output}"
                ),
            }
        ]
    })
    reader_output = _to_text(reader_result["messages"][-1].content)
    timings["read"] = time.time() - t0
    notify("read", "done", reader_output)

    # Step 3: Generate the research report
    notify("write", "start")
    t0 = time.time()
    report = write_chain.invoke({
        "topic": topic,
        "research_data": reader_output,
    })
    timings["write"] = time.time() - t0
    notify("write", "done", report)

    # Step 4: Review the report
    notify("critic", "start")
    t0 = time.time()
    critique = critic_chain.invoke({
        "topic": topic,
        "research_data": reader_output,
        "report": report,
    })
    timings["critic"] = time.time() - t0
    notify("critic", "done", critique)

    return {
        "topic": topic,
        "search_output": search_output,
        "reader_output": reader_output,
        "report": report,
        "critique": critique,
        "timings": timings,
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

STEP_LABELS = {
    "search": "🔎 Searching the web",
    "read": "📖 Reading and analyzing sources",
    "write": "✍️ Writing the report",
    "critic": "🧐 Reviewing the report",
}


def research_ui():
    st.set_page_config(
        page_title="Multi-Agent Research Assistant",
        page_icon="🧠",
        layout="wide",
    )

    # ---- Session state ----
    if "result" not in st.session_state:
        st.session_state.result = None
    if "error" not in st.session_state:
        st.session_state.error = None
    if "topic_input" not in st.session_state:
        st.session_state.topic_input = ""

    # ---- Sidebar ----
    with st.sidebar:
        st.header("🧠 Research Assistant")
        st.markdown(
            """
            **How it works**

            1. 🔎 **Search Agent** finds relevant sources
            2. 📖 **Reader Agent** scrapes and analyzes them
            3. ✍️ **Writer** produces a structured report
            4. 🧐 **Critic** reviews the report
            """
        )
        st.divider()
        st.caption("Try an example topic:")
        examples = [
            "Agentic AI",
            "Retrieval-Augmented Generation (RAG)",
            "Model Context Protocol (MCP)",
            "Vector databases",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
                st.session_state.topic_input = ex
                st.rerun()

        st.divider()
        if st.button("🗑️ Clear results", use_container_width=True):
            st.session_state.result = None
            st.session_state.error = None
            st.rerun()

    # ---- Header ----
    st.title("🧠 Multi-Agent Research Assistant")
    st.caption("Search → Read → Write → Critique, powered by LangChain agents.")

    # ---- Input ----
    topic = st.text_input(
        "Research topic",
        key="topic_input",
        placeholder="e.g. Agentic AI, quantum computing, RAG systems...",
    )

    run_clicked = st.button("🚀 Run Research", type="primary")

    # ---- Run pipeline ----
    if run_clicked:
        if not topic.strip():
            st.warning("Please enter a research topic first.")
        else:
            st.session_state.result = None
            st.session_state.error = None

            with st.status("Running research pipeline...", expanded=True) as status:

                def on_step(step, state, output=None):
                    label = STEP_LABELS[step]
                    if state == "start":
                        st.write(f"{label}...")
                        status.update(label=f"{label}...")
                    else:
                        st.write(f"✅ {label} — done")

                try:
                    st.session_state.result = run_research_pipeline(
                        topic.strip(), on_step=on_step
                    )
                    status.update(
                        label="Research complete!",
                        state="complete",
                        expanded=False,
                    )
                except Exception as e:
                    st.session_state.error = str(e)
                    status.update(label="Pipeline failed", state="error")

    # ---- Error display ----
    if st.session_state.error:
        st.error(f"Something went wrong: {st.session_state.error}")

    # ---- Results ----
    result = st.session_state.result
    if result:
        st.divider()
        st.subheader(f"Results: {result['topic']}")

        total_time = sum(result["timings"].values())
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Search", f"{result['timings']['search']:.1f}s")
        c2.metric("Read", f"{result['timings']['read']:.1f}s")
        c3.metric("Write", f"{result['timings']['write']:.1f}s")
        c4.metric("Critic", f"{result['timings']['critic']:.1f}s")
        c5.metric("Total", f"{total_time:.1f}s")

        tab_report, tab_critic, tab_reader, tab_search = st.tabs(
            ["📄 Report", "🧐 Critic Feedback", "📖 Reader Findings", "🔎 Search Results"]
        )

        with tab_report:
            st.markdown(result["report"])
            st.download_button(
                "⬇️ Download report (.md)",
                data=result["report"],
                file_name="research_report.md",
                mime="text/markdown",
            )

        with tab_critic:
            st.markdown(result["critique"])
            st.download_button(
                "⬇️ Download critique (.md)",
                data=result["critique"],
                file_name="critic_feedback.md",
                mime="text/markdown",
            )

        with tab_reader:
            st.markdown(result["reader_output"])

        with tab_search:
            st.markdown(result["search_output"])