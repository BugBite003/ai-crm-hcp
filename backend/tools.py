import pymysql
import json
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="gemma2-9b-it",
    api_key=os.getenv("GROQ_API_KEY")
)


def get_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", "root"),
        database=os.getenv("DB_NAME", "hcp_crm")
    )


# ─────────────────────────────────────────────
# TOOL 1: Log Interaction
# Uses Groq gemma2-9b-it to extract structured
# entities from free-text, then saves to DB.
# ─────────────────────────────────────────────
def log_interaction_tool(data: dict) -> dict:
    """
    Accepts either structured data (from form) or
    raw free-text (from chat). If 'raw_text' key is
    present the LLM extracts entities first.
    """
    if "raw_text" in data:
        prompt = f"""
You are a CRM assistant for a pharmaceutical field rep.
Extract the following fields from the interaction description below.
Return ONLY a valid JSON object with these exact keys:
hcp_name, interaction_type, date (YYYY-MM-DD), time (HH:MM:SS),
attendees, topics_discussed, sentiment (Positive/Neutral/Negative),
outcomes, follow_up

If a field is not mentioned, use a sensible default (e.g. today's date,
"Meeting" for type, "Neutral" for sentiment, empty string otherwise).

Interaction description:
\"\"\"{data['raw_text']}\"\"\"

Return only the JSON, no explanation, no markdown fences.
"""
        response = llm.invoke(prompt).content.strip()
        # Strip markdown fences if model adds them
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        try:
            extracted = json.loads(response)
        except json.JSONDecodeError:
            extracted = {
                "hcp_name": "Unknown HCP",
                "interaction_type": "Meeting",
                "date": "2026-04-27",
                "time": "12:00:00",
                "attendees": "",
                "topics_discussed": data["raw_text"],
                "sentiment": "Neutral",
                "outcomes": "",
                "follow_up": ""
            }
        data = extracted

    conn = get_connection()
    cursor = conn.cursor()
    query = """
        INSERT INTO interactions
        (hcp_name, interaction_type, date, time, attendees,
         topics_discussed, sentiment, outcomes, follow_up)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (
        data.get("hcp_name", ""),
        data.get("interaction_type", "Meeting"),
        data.get("date", "2026-04-27"),
        data.get("time", "12:00:00"),
        data.get("attendees", ""),
        data.get("topics_discussed", ""),
        data.get("sentiment", "Neutral"),
        data.get("outcomes", ""),
        data.get("follow_up", "")
    ))
    conn.commit()
    interaction_id = cursor.lastrowid
    conn.close()
    return {
        "status": "success",
        "message": "Interaction logged successfully",
        "interaction_id": interaction_id,
        "extracted_data": data
    }


# ─────────────────────────────────────────────
# TOOL 2: Edit Interaction
# Fetches the existing record, applies patch,
# optionally re-summarizes notes with LLM.
# ─────────────────────────────────────────────
def edit_interaction_tool(interaction_id: int, updates: dict) -> dict:
    """
    Updates specific fields of an existing interaction.
    If 'notes_patch' is provided, the LLM merges the new
    text with existing topics_discussed.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Fetch current record
    cursor.execute(
        "SELECT * FROM interactions WHERE id = %s", (interaction_id,)
    )
    row = cursor.fetchone()
    if not row:
        conn.close()
        return {"status": "error", "message": f"Interaction {interaction_id} not found"}

    cols = [desc[0] for desc in cursor.description]
    current = dict(zip(cols, row))

    # If there's a free-text patch, use LLM to merge it
    if "notes_patch" in updates:
        merge_prompt = f"""
You are editing a CRM interaction note.
Existing topics discussed: \"{current['topics_discussed']}\"
New information to merge: \"{updates['notes_patch']}\"
Write a concise merged summary combining both. Return only the summary text.
"""
        merged = llm.invoke(merge_prompt).content.strip()
        updates["topics_discussed"] = merged
        del updates["notes_patch"]

    # Build dynamic UPDATE
    allowed_fields = {
        "hcp_name", "interaction_type", "date", "time",
        "attendees", "topics_discussed", "sentiment",
        "outcomes", "follow_up"
    }
    filtered = {k: v for k, v in updates.items() if k in allowed_fields}
    if not filtered:
        conn.close()
        return {"status": "error", "message": "No valid fields to update"}

    set_clause = ", ".join(f"{k} = %s" for k in filtered)
    values = list(filtered.values()) + [interaction_id]
    cursor.execute(f"UPDATE interactions SET {set_clause} WHERE id = %s", values)
    conn.commit()
    conn.close()

    return {
        "status": "success",
        "message": f"Interaction {interaction_id} updated",
        "updated_fields": list(filtered.keys())
    }


# ─────────────────────────────────────────────
# TOOL 3: Get HCP Profile
# Returns full interaction history for an HCP.
# ─────────────────────────────────────────────
def get_hcp_profile_tool(hcp_name: str) -> dict:
    """
    Retrieves all interactions for a given HCP
    and returns a summary with history.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM interactions WHERE hcp_name = %s ORDER BY date DESC",
        (hcp_name,)
    )
    rows = cursor.fetchall()
    cols = [desc[0] for desc in cursor.description]
    conn.close()

    if not rows:
        return {
            "status": "not_found",
            "message": f"No interactions found for {hcp_name}",
            "interactions": []
        }

    interactions = [dict(zip(cols, row)) for row in rows]
    # Convert date/time objects to strings for JSON serialisation
    for item in interactions:
        for key, val in item.items():
            item[key] = str(val) if not isinstance(val, (str, int, float, type(None))) else val

    return {
        "status": "success",
        "hcp_name": hcp_name,
        "total_interactions": len(interactions),
        "interactions": interactions
    }


# ─────────────────────────────────────────────
# TOOL 4: Suggest Follow-up (LLM-powered)
# Generates intelligent next steps based on
# the latest interaction data.
# ─────────────────────────────────────────────
def suggest_followup_tool(interaction_data: dict) -> dict:
    """
    Uses gemma2-9b-it to analyse the interaction
    and generate 3 specific, actionable follow-ups.
    """
    prompt = f"""
You are an AI assistant for a pharmaceutical field representative.
Based on the following HCP interaction, suggest exactly 3 specific,
actionable follow-up actions. Be concise and professional.

HCP: {interaction_data.get('hcp_name', 'Unknown')}
Interaction type: {interaction_data.get('interaction_type', 'Meeting')}
Topics discussed: {interaction_data.get('topics_discussed', '')}
Sentiment observed: {interaction_data.get('sentiment', 'Neutral')}
Outcomes: {interaction_data.get('outcomes', '')}

Return ONLY a JSON object with this structure:
{{
  "suggestions": [
    "Follow-up action 1",
    "Follow-up action 2",
    "Follow-up action 3"
  ],
  "priority": "High" | "Medium" | "Low",
  "reasoning": "One sentence why"
}}
Return only the JSON, no markdown fences.
"""
    response = llm.invoke(prompt).content.strip()
    if response.startswith("```"):
        response = response.split("```")[1]
        if response.startswith("json"):
            response = response[4:]
    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        result = {
            "suggestions": [
                "Schedule follow-up meeting within 7 days",
                "Send relevant product literature",
                "Update HCP record with discussion outcomes"
            ],
            "priority": "Medium",
            "reasoning": "Standard follow-up based on interaction"
        }
    return {"status": "success", **result}


# ─────────────────────────────────────────────
# TOOL 5: Analyse Sentiment
# Uses LLM to infer and explain HCP sentiment
# from the interaction notes.
# ─────────────────────────────────────────────
def analyse_sentiment_tool(notes: str) -> dict:
    """
    Uses gemma2-9b-it to perform sentiment analysis
    on interaction notes and return label + explanation.
    """
    prompt = f"""
You are analysing a pharmaceutical sales interaction note.
Classify the HCP's sentiment and explain briefly.

Interaction notes:
\"\"\"{notes}\"\"\"

Return ONLY a JSON object:
{{
  "sentiment": "Positive" | "Neutral" | "Negative",
  "confidence": "High" | "Medium" | "Low",
  "explanation": "One sentence explaining the sentiment",
  "key_signals": ["signal 1", "signal 2"]
}}
Return only the JSON, no markdown fences.
"""
    response = llm.invoke(prompt).content.strip()
    if response.startswith("```"):
        response = response.split("```")[1]
        if response.startswith("json"):
            response = response[4:]
    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        result = {
            "sentiment": "Neutral",
            "confidence": "Low",
            "explanation": "Could not analyse sentiment from provided notes",
            "key_signals": []
        }
    return {"status": "success", **result}