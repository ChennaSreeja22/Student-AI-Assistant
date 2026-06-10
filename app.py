import streamlit as st
import fitz
from groq import Groq
from rag_utils import chunk_text, store_chunks, retrieve_chunks

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

def generate_quiz(client, text, difficulty, quiz_type):
    if quiz_type == "MCQ":
        prompt = f"""Generate 5 {difficulty} level multiple choice questions from this material.
Format each as:
Q1. Question
a) option
b) option
c) option
d) option
Answer: correct option

Material: {text}"""
    else:
        prompt = f"""Generate 5 {difficulty} level theory questions from this material.
These should be short answer or descriptive questions.
Format each as:
Q1. Question
Answer: detailed answer

Material: {text}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def extract_key_points(client, text):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Extract key points as a numbered list from this: " + text}]
    )
    return response.choices[0].message.content

def answer_question(client, question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = f"""Answer the following question based only on the provided context.
If the answer is not in the context, say "I could not find this in the document."

Context:
{context}

Question: {question}"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

st.title("Student AI Assistant")
st.write("Upload a PDF to get a summary, quiz, key points or chat with your document.")

api_key = st.text_input("Enter your Groq API Key", type="password")
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and api_key:
    with st.spinner("Reading PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    st.success("PDF loaded successfully!")

    tab1, tab2, tab3, tab4 = st.tabs(["Summarize", "Quiz", "Key Points", "Chat with PDF"])

    with tab1:
        if st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                client = configure_groq(api_key)
                summary = summarize_text(client, text)
            st.subheader("Summary")
            st.write(summary)

    with tab2:
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
        quiz_type = st.selectbox("Quiz Type", ["MCQ", "Theory"])
        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz..."):
                client = configure_groq(api_key)
                quiz = generate_quiz(client, text, difficulty, quiz_type)
            st.subheader(f"{difficulty} {quiz_type} Quiz")
            st.write(quiz)
            st.download_button(
                label="Download Quiz",
                data=quiz,
                file_name="quiz.txt",
                mime="text/plain"
            )

    with tab3:
        if st.button("Extract Key Points"):
            with st.spinner("Extracting key points..."):
                client = configure_groq(api_key)
                points = extract_key_points(client, text)
            st.subheader("Key Points")
            st.write(points)

    with tab4:
        st.subheader("Chat with your PDF")
        with st.spinner("Processing PDF for chat..."):
            chunks = chunk_text(text)
            index, chunks = store_chunks(chunks)
        st.success(f"PDF processed into {len(chunks)} chunks.")

        question = st.text_input("Ask a question from your PDF")
        if st.button("Get Answer") and question:
            with st.spinner("Searching document..."):
                relevant_chunks = retrieve_chunks(question, index, chunks)
                client = configure_groq(api_key)
                answer = answer_question(client, question, relevant_chunks)
            st.subheader("Answer")
            st.write(answer)