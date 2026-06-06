import streamlit as st
import fitz
from groq import Groq

def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def configure_groq(api_key):
    return Groq(api_key=api_key)

def summarize_text(client, text):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Summarize this study material clearly: " + text}]
    )
    return response.choices[0].message.content

def generate_quiz(client, text):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Generate 5 MCQ questions from this material: " + text}]
    )
    return response.choices[0].message.content

def extract_key_points(client, text):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Extract key points as a numbered list from this: " + text}]
    )
    return response.choices[0].message.content

st.title("Student AI Assistant")
st.write("Upload a PDF to get a summary and quiz questions.")

api_key = st.text_input("Enter your Groq API Key", type="password")
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and api_key:
    with st.spinner("Reading PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    st.success("PDF loaded successfully!")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Summarize"):
            with st.spinner("Generating summary..."):
                client = configure_groq(api_key)
                summary = summarize_text(client, text)
            st.subheader("Summary")
            st.write(summary)

    with col2:
        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz..."):
                client = configure_groq(api_key)
                quiz = generate_quiz(client, text)
            st.subheader("Quiz")
            st.write(quiz)

    with col3:
        if st.button("Key Points"):
            with st.spinner("Extracting key points..."):
                client = configure_groq(api_key)
                points = extract_key_points(client, text)
            st.subheader("Key Points")
            st.write(points)
            