content = """from groq import Groq

def configure_groq(api_key):
    client = Groq(api_key=api_key)
    return client

def summarize_text(client, text):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": f"Summarize the following study material clearly and concisely:\\n\\n{text}"}
        ]
    )
    return response.choices[0].message.content

def generate_quiz(client, text):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": f"Generate 5 multiple choice questions from the following study material.\\n\\nMaterial:\\n{text}"}
        ]
    )
    return response.choices[0].message.content

def extract_key_points(client, text):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": f"Extract the most important key points as a numbered list from:\\n\\n{text}"}
        ]
    )
    return response.choices[0].message.content
"""

with open("llm_utils.py", "w", encoding="utf-8") as f:
    f.write(content)

print("llm_utils.py rewritten successfully")
    