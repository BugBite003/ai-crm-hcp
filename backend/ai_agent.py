import json
import os
from typing import TypedDict, Literal

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from tools import (
    log_interaction_tool,
    edit_interaction_tool,
    get_hcp_profile_tool,
    suggest_followup_tool,
    analyse_sentiment_tool,
)

load_dotenv()

# ── LLM (primary: gemma2-9b-it as required) ──────────────────────────────────
# ── LLM configuration ─────────────────────────────────────────────────────────
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_MODEL = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
llm = ChatGroq(
    model="GROQ_MODEL",
    api_key=os.getenv("GROQ_API_KEY")
)

# ── Agent State ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    input: str
    intent: str          # classified intent
    tool_result: dict    # result from whichever tool ran
    output: str          # final human-readable response


# ── Node 1: Classify Intent ───────────────────────────────────────────────────
def classify_intent_node(state: AgentState) -> AgentState:
    """
    Uses the configured Groq model to decide which tool to invoke
    based on the user's natural-language input.
    """
    prompt = f"""
You are a routing agent for a pharmaceutical CRM system.
Classify the user's intent into EXACTLY one of these labels:
  log_interaction   - user wants to log or record a new HCP meeting/interaction
  edit_interaction  - user wants to update or modify an existing interaction
  get_hcp_profile   - user wants to see history or profile for an HCP
  suggest_followup  - user wants follow-up recommendations after a meeting
  analyse_sentiment - user wants sentiment analysis of interaction notes
  unknown           - none of the above

User message: "{state['input']}"

Return ONLY the label, nothing else.
"""
    intent = llm.invoke(prompt).content.strip().lower()
    valid = {
        "log_interaction", "edit_interaction", "get_hcp_profile",
        "suggest_followup", "analyse_sentiment"
    }
    if intent not in valid:
        intent = "log_interaction"  # safe default for chat logging

    return {**state, "intent": intent}


# ── Node 2a: Log Interaction ──────────────────────────────────────────────────
def log_node(state: AgentState) -> AgentState:
    result = log_interaction_tool({"raw_text": state["input"]})
    return {**state, "tool_result": result}


# ── Node 2b: Edit Interaction ─────────────────────────────────────────────────
def edit_node(state: AgentState) -> AgentState:
    """
    Extracts interaction_id and patch text from the user message.
    Expected pattern: 'update interaction 5 ...' or 'edit #5 ...'
    """
    import re
    match = re.search(r"#?(\d+)", state["input"])
    interaction_id = int(match.group(1)) if match else 1
    result = edit_interaction_tool(
        interaction_id=interaction_id,
        updates={"notes_patch": state["input"]}
    )
    return {**state, "tool_result": result}


# ── Node 2c: Get HCP Profile ──────────────────────────────────────────────────
def get_hcp_node(state: AgentState) -> AgentState:
    """
    Extracts HCP name from the user message using the LLM.
    """
    prompt = f"""
Extract the HCP (doctor/healthcare professional) name from this message.
Return only the name, nothing else.
If no name found return "Unknown".

Message: "{state['input']}"
"""
    hcp_name = llm.invoke(prompt).content.strip()
    result = get_hcp_profile_tool(hcp_name)
    return {**state, "tool_result": result}


# ── Node 2d: Suggest Follow-up ────────────────────────────────────────────────
def suggest_node(state: AgentState) -> AgentState:
    # Use the input text as the interaction context
    interaction_data = {
        "hcp_name": "HCP from conversation",
        "interaction_type": "Meeting",
        "topics_discussed": state["input"],
        "sentiment": "Neutral",
        "outcomes": ""
    }
    result = suggest_followup_tool(interaction_data)
    return {**state, "tool_result": result}


# ── Node 2e: Analyse Sentiment ────────────────────────────────────────────────
def sentiment_node(state: AgentState) -> AgentState:
    result = analyse_sentiment_tool(state["input"])
    return {**state, "tool_result": result}


# ── Node 3: Format Response ───────────────────────────────────────────────────
def format_response_node(state: AgentState) -> AgentState:
    """
    Converts the raw tool_result dict into a friendly
    human-readable response using the configured Groq model.
    """
    prompt = f"""
You are a helpful CRM assistant for pharmaceutical field representatives.
Convert this tool result into a brief, friendly, professional response.
Be specific and helpful. Use 2-4 sentences.

Tool used: {state['intent']}
Result: {json.dumps(state['tool_result'])}

Return only the response text.
"""
    output = llm.invoke(prompt).content.strip()
    return {**state, "output": output}


# ── Router ────────────────────────────────────────────────────────────────────
def route_to_tool(state: AgentState) -> Literal[
    "log_node", "edit_node", "get_hcp_node", "suggest_node", "sentiment_node"
]:
    routes = {
        "log_interaction": "log_node",
        "edit_interaction": "edit_node",
        "get_hcp_profile": "get_hcp_node",
        "suggest_followup": "suggest_node",
        "analyse_sentiment": "sentiment_node",
    }
    return routes.get(state["intent"], "log_node")


# ── Build LangGraph ───────────────────────────────────────────────────────────
graph = StateGraph(AgentState)

graph.add_node("classify_intent", classify_intent_node)
graph.add_node("log_node", log_node)
graph.add_node("edit_node", edit_node)
graph.add_node("get_hcp_node", get_hcp_node)
graph.add_node("suggest_node", suggest_node)
graph.add_node("sentiment_node", sentiment_node)
graph.add_node("format_response", format_response_node)

graph.set_entry_point("classify_intent")

graph.add_conditional_edges(
    "classify_intent",
    route_to_tool,
    {
        "log_node": "log_node",
        "edit_node": "edit_node",
        "get_hcp_node": "get_hcp_node",
        "suggest_node": "suggest_node",
        "sentiment_node": "sentiment_node",
    }
)

# All tool nodes flow into format_response
for node in ["log_node", "edit_node", "get_hcp_node", "suggest_node", "sentiment_node"]:
    graph.add_edge(node, "format_response")

graph.add_edge("format_response", END)

app_graph = graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────
def run_agent(text: str) -> dict:
    """
    Entry point called by FastAPI /chat endpoint.
    Returns output string + structured tool_result.
    """
    initial_state: AgentState = {
        "input": text,
        "intent": "",
        "tool_result": {},
        "output": ""
    }
    final_state = app_graph.invoke(initial_state)
    return {
        "response": final_state["output"],
        "intent": final_state["intent"],
        "tool_result": final_state["tool_result"]
    }   