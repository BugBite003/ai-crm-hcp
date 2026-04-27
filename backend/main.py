from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pymysql
import os
from dotenv import load_dotenv

from ai_agent import run_agent
from tools import (
    log_interaction_tool,
    edit_interaction_tool,
    get_hcp_profile_tool,
    suggest_followup_tool,
    analyse_sentiment_tool,
)

load_dotenv()

app = FastAPI(title="AI-First CRM HCP Module", version="1.0.0")

default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
allowed_origins = os.getenv("CORS_ALLOW_ORIGINS")
if allowed_origins:
    cors_origins = [origin.strip() for origin in allowed_origins.split(",") if origin.strip()]
else:
    cors_origins = default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── DB helper ─────────────────────────────────────────────────────────────────
def get_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", "root"),
        database=os.getenv("DB_NAME", "hcp_crm")
    )


# ── Pydantic Models ───────────────────────────────────────────────────────────
class InteractionForm(BaseModel):
    hcp_name: str
    interaction_type: str
    date: str
    time: str
    attendees: Optional[str] = ""
    topics_discussed: Optional[str] = ""
    sentiment: Optional[str] = "Neutral"
    outcomes: Optional[str] = ""
    follow_up: Optional[str] = ""


class EditRequest(BaseModel):
    interaction_id: int
    updates: dict


class ChatMessage(BaseModel):
    message: str


class SentimentRequest(BaseModel):
    notes: str


class FollowUpRequest(BaseModel):
    hcp_name: str
    interaction_type: Optional[str] = "Meeting"
    topics_discussed: Optional[str] = ""
    sentiment: Optional[str] = "Neutral"
    outcomes: Optional[str] = ""


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"status": "AI-First CRM API running", "version": "1.0.0"}


# ── Tool 1: Log Interaction (form-based) ──────────────────────────────────────
@app.post("/log")
def log_interaction(data: InteractionForm):
    """Logs a structured interaction from the form UI."""
    result = log_interaction_tool(data.model_dump())
    if result["status"] != "success":
        raise HTTPException(status_code=500, detail=result.get("message"))
    return result


# ── Tool 2: Edit Interaction ──────────────────────────────────────────────────
@app.put("/interaction/{interaction_id}")
def edit_interaction(interaction_id: int, updates: dict):
    """Updates fields of an existing interaction."""
    result = edit_interaction_tool(interaction_id, updates)
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    return result


# ── Tool 3: Get HCP Profile ───────────────────────────────────────────────────
@app.get("/hcp/{hcp_name}")
def get_hcp(hcp_name: str):
    """Returns interaction history for a given HCP."""
    result = get_hcp_profile_tool(hcp_name)
    return result


# ── Tool 4: Suggest Follow-up ─────────────────────────────────────────────────
@app.post("/suggest-followup")
def suggest_followup(data: FollowUpRequest):
    """Generates AI-powered follow-up suggestions using the configured Groq model."""
    result = suggest_followup_tool(data.model_dump())
    return result


# ── Tool 5: Analyse Sentiment ─────────────────────────────────────────────────
@app.post("/analyse-sentiment")
def analyse_sentiment(data: SentimentRequest):
    """Analyses HCP sentiment from interaction notes using the configured Groq model.."""
    result = analyse_sentiment_tool(data.notes)
    return result


# ── AI Chat (LangGraph agent) ─────────────────────────────────────────────────
@app.post("/chat")
def chat(body: ChatMessage):
    """
    Main LangGraph agent endpoint. Classifies intent and
    routes to the appropriate tool automatically.
    """
    try:
        result = run_agent(body.message)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(exc)}")


# ── List all interactions ─────────────────────────────────────────────────────
@app.get("/interactions")
def list_interactions():
    """Returns all logged interactions (for UI display)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM interactions ORDER BY id DESC LIMIT 50")
    rows = cursor.fetchall()
    cols = [desc[0] for desc in cursor.description]
    conn.close()
    interactions = []
    for row in rows:
        item = {}
        for key, val in zip(cols, row):
            item[key] = str(val) if not isinstance(val, (str, int, float, type(None))) else val
        interactions.append(item)
    return {"interactions": interactions}