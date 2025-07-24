import streamlit as st
import fitz 
import google.generativeai as genai
import numpy as np
import faiss
import json
import re
from sentence_transformers import SentenceTransformer

genai.configure(api_key="api key here")
model = genai.GenerativeModel("gemini-1.5-pro-latest")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

dimension = 384
index = faiss.IndexFlatL2(dimension)
resume_store = {}


def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "".join([page.get_text() for page in doc])


def parse_resume_with_rag(text):
    query_embedding = embedding_model.encode([text])
    if index.ntotal > 0:
        _, ids = index.search(query_embedding, k=3)
        context = "\n".join([resume_store[i] for i in ids[0] if i in resume_store])
    else:
        context = ""

    prompt = f"""
You are an AI Resume Parser. Extract the following fields in JSON format:
{{
    "name": "",
    "contact": {{"phone": "", "email": ""}},
    "education": [{{"degree": "", "institution": "", "year": ""}}],
    "experience": [{{"title": "", "company": "", "duration": "", "responsibilities": [""]}}],
    "skills": ["skill1", "skill2"],
    "projects": ["project1", "project2"],
    "internships": [
        {{
            "title": "",
            "company": "",
            "duration": "",
            "work": ""
        }}
    ]
}}
Resume:
{text}
Context from similar resumes:
{context}
"""


    response = model.generate_content(prompt)
    match = re.search(r"\{.*\}", response.text, re.DOTALL)
    return json.loads(match.group(0)) if match else {"error": "Parsing failed"}


def score_resume(text, jd):
    prompt = f"""
    You are an AI Resume Evaluator.

    Evaluate the resume below against the given job description and provide:
    - A relevance score (0-100).
    - A brief feedback summary.
    - A chain-of-thought explanation explaining how you arrived at the score, comparing skills, experience, and relevance.

    Return only JSON in this format:
    {{
        "score": integer,
        "feedback": "short comment",
        "chain_of_thought": "step-by-step reasoning"
    }}

    Job Description:
    {jd}

    Resume:
    {text}
    """

    response = model.generate_content(prompt)
    match = re.search(r"\{.*\}", response.text, re.DOTALL)
    return json.loads(match.group(0)) if match else {"error": "Scoring failed"}


def generate_assessment(skills, projects, internships):
    projects_text = " ".join([p if isinstance(p, str) else " ".join([str(v) for v in p.values()]) for p in projects])
    
    prompt = f"""
    You are an AI resume parser and verifier designed to analyze job applicants' resumes and assess their fit for specific job descriptions. Your task is to generate relevant assessment questions based on the following parsed data.

    Resume Information:
    Skills: {skills}
    Projects: {projects_text}
    Internships: {internships}

    Generate an assessment containing:
    - 3 conceptual multiple-choice questions (MCQs) based on the candidate's skills.
    - 2 fill-in-the-blank questions (each with one blank) based on technology terms, definitions, or usage from the candidate's project or internship experience.
    - 2 partial code completion questions that require the candidate to **complete an entire function**.
    - Ensure MCQs include conceptual traps or scenario-based options.
    - Ensure fill-in-the-blanks use technical usage in context (not definitions alone).
    - For the fill in the blanks questions make sure there are new questions generated everytime which is based on the projects and skills only.
    - Ensure code completions use compound logic (loop + condition + transformation).
    - For the code filling questions make sure there are new questions generated everytime which is based on the projects and skills only.
    - The level of the questions should be intermediate to hard.
    - Use edge cases, realistic workplace scenarios, and knowledge of advanced tools/concepts in the candidate's field.



    Partial Code Completion Guidelines:
    - Do NOT format as fill-in-the-blanks.
    - Present a function header and description with a comment like "# TODO: Complete this function".
    - Ensure the candidate must implement meaningful intermediate-level logic.
    - Focus on concepts like loops, string/list manipulation, simple algorithms, filtering, or working with mock data/APIs.
    - Use information from the candidate’s resume (skills, projects, internships) to inspire the function's purpose.
    - Variable names and function names may vary; the answer will be validated based on logical structure, not syntax.

    Return the output in the following JSON structure:
    {{
        "mcqs": [
            {{
                "question": "...",
                "options": ["..."],
                "answer": "..."
            }}
        ],
        "fill_in_the_blanks": [
            {{
                "question": "... ___ ...",
                "answer": "..."
            }}
        ],
        "code_completion": [
            {{
                "question": "def function_name(...):\\n    # TODO: Complete this function",
                "answer": "def function_name(...):\\n    line1\\n    line2\\n    line3\\n    line4"
            }}
        ]
    }}
    """
    
    response = model.generate_content(prompt)
    match = re.search(r"\{.*\}", response.text, re.DOTALL)
    return json.loads(match.group(0)) if match else {"error": "Assessment generation failed"}



st.set_page_config("Resume Parser & Assessment", layout="wide")
st.title("📄 Resume Parser & AI-Powered Assessment")

jd = st.text_area("Paste the Job Description here:")
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)
    resume_embedding = embedding_model.encode([resume_text])
    index.add(np.array(resume_embedding, dtype=np.float32))
    resume_store[index.ntotal - 1] = resume_text

    parsed_data = parse_resume_with_rag(resume_text)
    st.subheader("📋 Extracted Resume Info")
    st.json(parsed_data)

    if jd:
        score_data = score_resume(resume_text, jd)
        st.metric("🎯 Resume Relevance Score", score_data.get("score", 0))
        st.success(score_data.get("feedback", ""))
        with st.expander("🧠 Chain of Thought Explanation"):
            st.markdown(score_data.get("chain_of_thought", "No explanation available."))


        if score_data["score"] > 40:
            with st.spinner("Generating assessment..."):
                questions = generate_assessment(
                    parsed_data.get("skills", []),
                    parsed_data.get("projects", []),
                    parsed_data.get("internships", [])
                )
                st.session_state["assessment_data"] = questions
                st.success("✅ Assessment Ready!")
                st.page_link("pages/assessment.py", label="Start Assessment", icon="📝")
        else:
            st.warning("Score is below threshold. Improve resume and try again.")
