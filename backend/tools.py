import pymysql

def get_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="root",
        database="hcp_crm"
    )

# TOOL 1: Log Interaction
def log_interaction_tool(data):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    INSERT INTO interactions 
    (hcp_name, interaction_type, date, time, notes, sentiment, follow_up)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    cursor.execute(query, (
        data.get("hcp_name"),
        data.get("interaction_type"),
        data.get("date"),
        data.get("time"),
        data.get("notes"),
        data.get("sentiment"),
        data.get("follow_up")
    ))

    conn.commit()
    conn.close()

    return "Interaction logged successfully"


# TOOL 2: Edit Interaction
def edit_interaction_tool(id, notes):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("UPDATE interactions SET notes=%s WHERE id=%s", (notes, id))

    conn.commit()
    conn.close()

    return "Interaction updated"


# TOOL 3: Get HCP Details
def get_hcp_tool(name):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM interactions WHERE hcp_name=%s", (name,))
    data = cursor.fetchall()

    conn.close()
    return str(data)


# TOOL 4: Suggest Follow-up
def suggest_followup_tool():
    return "Suggested: Schedule follow-up within 7 days"


# TOOL 5: Summarize
def summarize_tool(text):
    return f"Summary: {text[:100]}"