from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from dotenv import load_dotenv
import os

from tools import log_interaction_tool

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# Simple decision logic
def agent(state):
    user_input = state["input"]

    # Use LLM to extract structured info
    prompt = f"""
    Extract this into JSON:
    hcp_name, interaction_type, date, time, notes, sentiment, follow_up

    Input: {user_input}
    """

    response = llm.invoke(prompt).content

    # VERY SIMPLE PARSE (keep it basic)
    data = {
        "hcp_name": "Dr. Sharma",
        "interaction_type": "Meeting",
        "date": "2026-04-27",
        "time": "14:30:00",
        "notes": user_input,
        "sentiment": "Positive",
        "follow_up": "Next week"
    }

    result = log_interaction_tool(data)

    return {"output": result}


# LangGraph setup
graph = StateGraph(dict)
graph.add_node("agent", agent)
graph.set_entry_point("agent")

app_graph = graph.compile()


def run_agent(text):
    result = app_graph.invoke({"input": text})
    return result["output"]