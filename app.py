import os
import streamlit as st
import fitz
import datetime
import time
from dotenv import load_dotenv
from groq import Groq
from rag_utils import chunk_text_with_metadata, store_chunks, retrieve_chunks_with_metadata
from collections import defaultdict
load_dotenv()

try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = os.getenv("GROQ_API_KEY")

def extract_text_from_pdf(uploaded_file):
    text = ""

    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        page_count = len(doc)

        for page in doc:
            text += page.get_text()

    return text, page_count

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

def answer_question(client, question, relevant_chunks, chat_history):
    context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
    messages = [
        {"role": "system", "content": f"""You are a helpful study assistant.
Answer questions based on the following document context.
If the answer is not in the context, say "I could not find this in the document."

Context:
{context}"""}
    ]
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return response.choices[0].message.content

def study_planner_agent(client, subjects, hours_per_day, exam_date, difficulty_levels):
    days_remaining = (exam_date - datetime.date.today()).days
    prompt = f"""You are an autonomous study planner agent.
Your job is to create a personalized day-by-day study plan.

Make intelligent decisions about:
- How many days to allocate per subject based on difficulty
- Which subjects to study first
- When to schedule revision sessions
- How to distribute hours per day

Student Information:
- Subjects: {subjects}
- Difficulty levels: {difficulty_levels}
- Available hours per day: {hours_per_day}
- Exam date: {exam_date}
- Days remaining: {days_remaining} days

Create a detailed day-by-day study plan. For each day specify:
- Date
- Subject to study
- Topics to cover
- Hours allocated
- Study strategy (new topic / revision / practice problems)

End with a summary of your planning decisions and reasoning."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

st.set_page_config(page_title="Student AI Assistant", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .main {background-color: #f5f7fa;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: bold;
    }
    .stSuccess {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("📚 Student AI Assistant")
    st.write("---")

    uploaded_files = st.file_uploader(
        "Upload your PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    doc_info = st.empty()
    st.write("---")
    st.markdown("""
    ### Features
    | Feature | Status |
    |---------|--------|
    | Summarize PDF | ✅ |
    | MCQ & Theory Quiz | ✅ |
    | Quiz Difficulty Levels | ✅ |
    | Download Quiz | ✅ |
    | Key Points Extraction | ✅ |
    | Chat with PDF | ✅ |
    | Source Citations | ✅ |
    | Conversation Memory | ✅ |
    | Multi-PDF Support | ✅ |
    | RAG Pipeline (FAISS) | ✅ |
    | Study Planner Agent | ✅ |
    """)

st.title("📚 Student AI Assistant")
st.write("Upload one or more PDFs and start learning smarter.")

if api_key and api_key.startswith("gsk_"):
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Summarize", "🧠 Quiz", "🔑 Key Points", "💬 Chat with PDF", "📅 Study Planner"])

    with tab5:
        st.subheader("📅 Study Planner Agent")
        st.write("The agent will autonomously create a personalized study plan for you.")

        subjects_input = st.text_area("Enter your subjects (one per line)", placeholder="Mathematics\nPhysics\nChemistry")
        difficulty_input = st.text_area("Enter difficulty for each subject (one per line)", placeholder="Hard\nMedium\nEasy")
        hours_per_day = st.slider("Study hours per day", 1, 12, 4)
        exam_date = st.date_input("Exam date", min_value=datetime.date.today())

        if st.button("Generate Study Plan"):
            if subjects_input and difficulty_input:

                subjects = subjects_input.strip().split("\n")
                difficulty_levels = difficulty_input.strip().split("\n")

                if len(subjects) != len(difficulty_levels):
                    st.error("Please enter one difficulty level for each subject.")
                   
                try:
                    with st.spinner("Agent is creating your personalized study plan..."):
                        client = configure_groq(api_key)

                        plan = study_planner_agent(
                            client,
                            subjects,
                            hours_per_day,
                            exam_date,
                            difficulty_levels
                        )

                        st.subheader("📅 Your Personalized Study Plan")
                        st.write(plan)

                        st.download_button(
                            label="Download Study Plan",
                            data=plan,
                            file_name="study_plan.txt",
                            mime="text/plain"
                        )

                except Exception as e:
                    st.error(f"Unable to generate study plan.\n\n{e}")

                    st.subheader("Your Personalized Study Plan")
                    st.write(plan)
        
            else:
                st.warning("Please enter your subjects and difficulty levels.")

    if uploaded_files:

        st.session_state.pipeline_start = time.perf_counter()

        with st.spinner("Reading PDFs..."):
            all_chunks = []
            all_metadata = []
            pdf_info = []
            total_pages = 0

            for uploaded_file in uploaded_files:
                text, page_count = extract_text_from_pdf(uploaded_file)

                pdf_info.append({
                    "name": uploaded_file.name,
                    "pages": page_count
                })

                total_pages += page_count

                chunks, metadata = chunk_text_with_metadata(
                    text,
                    uploaded_file.name
                )

                all_chunks.extend(chunks)
                all_metadata.extend(metadata)
            index, all_chunks = store_chunks(all_chunks)
            text = " ".join(all_chunks)

            with doc_info.container():

                st.success("📄 Documents Loaded")

                st.metric("PDFs", len(uploaded_files))
                st.metric("Pages", total_pages)
                st.metric("Chunks", len(all_chunks))

                with st.expander("View Documents", expanded=False):
                    for pdf in pdf_info:
                        st.write(f"📄 {pdf['name']}")
                        st.caption(f"{pdf['pages']} pages")

        with tab1:
            if st.button("Generate Summary"):
                try:
                    with st.spinner("Generating summary..."):
                        client = configure_groq(api_key)
                        summary = summarize_text(client, text)

                    st.subheader("📄 AI Summary")
                    st.write(summary)

                    st.download_button(
                        label="⬇️ Download Summary",
                        data=summary,
                        file_name="summary.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"Unable to generate summary.\n\n{e}")


        with tab2:
            difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
            quiz_type = st.selectbox("Quiz Type", ["MCQ", "Theory"])
            if st.button("Generate Quiz"):                

                try:
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

                except Exception as e:
                    st.error(f"Unable to generate quiz.\n\n{e}")
                

        with tab3:
            if st.button("Extract Key Points"):

                try:
                    with st.spinner("Extracting key points..."):
                        client = configure_groq(api_key)
                        points = extract_key_points(client, text)

                    st.subheader("🔑 Key Points")
                    st.write(points)

                    st.download_button(
                       label="⬇️ Download Key Points",
                       data=points,
                       file_name="key_points.txt",
                       mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"Unable to extract key points.\n\n{e}")

                

        with tab4:
            st.subheader("Chat with your PDF")

            col1, col2 = st.columns([4, 1])

            with col2:
                if st.button("🗑️ Clear Chat"):
                   st.session_state.messages = []
                   st.rerun()

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            question = st.chat_input("Ask a question from your PDF...")
            if question:
                with st.chat_message("user"):
                    st.write(question)
                st.session_state.messages.append({
                    "role": "user",
                    "content": question
                })

                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Thinking..."):

                            start_time = time.perf_counter()
                            relevant_results = retrieve_chunks_with_metadata(
                                question,
                                index,
                                all_metadata
                            )

                            client = configure_groq(api_key)

                            answer = answer_question(
                                client,
                                question,
                                relevant_results,
                                st.session_state.messages
                            )

                            end_time = time.perf_counter()

                        rag_time = end_time - start_time   
                    except Exception as e:
                        st.error(f"Unable to answer your question.\n\n{e}")
                        answer = None
            
                if answer:
                    st.write(answer)

                    st.success(
                        f"⏱️ Core RAG pipeline: {rag_time:.2f} seconds"
                    )
                    st.write("---")
                    st.markdown("### 📚 Sources")

                    sources = defaultdict(set)

                    for result in relevant_results:
                        sources[result["file"]].add(result["page"])

                    for file, pages in sources.items():
                        pages = sorted(pages)

                        if len(pages) == 1:
                            st.caption(f"📄 **{file}**")
                            st.caption(f"Page: {pages[0]}")
                        else:
                            page_list = ", ".join(str(p) for p in pages)
                            st.caption(f"📄 **{file}**")
                            st.caption(f"Pages: {page_list}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                    with st.expander("📖 View Retrieved Context"):
                        for i, result in enumerate(relevant_results, start=1):
                            st.markdown(
                                f"**Chunk {i}**\n"
                                f"- File: {result['file']}\n"
                                f"- Page: {result['page']}"
                            )

                            st.write(result["text"])

                            if i != len(relevant_results):
                                st.divider()
                   
    else:
        with tab1:
            st.info("Please upload a PDF to use this feature.")
        with tab2:
            st.info("Please upload a PDF to use this feature.")
        with tab3:
            st.info("Please upload a PDF to use this feature.")
        with tab4:
            st.info("Please upload a PDF to use this feature.")

else:
    st.error(
    "Groq API key not found. Please add GROQ_API_KEY to your .env file."
)