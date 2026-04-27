from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from ai_agent import run_agent
import pymysql

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Interaction(BaseModel):
    hcp_name: str
    interaction_type: str
    date: str
    time: str
    notes: str
    sentiment: str
    follow_up: str

def get_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="root",
        database="hcp_crm"
    )

@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/log")
def log_interaction(data: Interaction):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    INSERT INTO interactions 
    (hcp_name, interaction_type, date, time, notes, sentiment, follow_up)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    cursor.execute(query, (
        data.hcp_name,
        data.interaction_type,
        data.date,
        data.time,
        data.notes,
        data.sentiment,
        data.follow_up
    ))

    conn.commit()
    conn.close()

    return {"message": "Saved successfully"}

@app.post("/chat")
def chat(input: dict):
    text = input.get("message")
    result = run_agent(text)
    return {"response": result}