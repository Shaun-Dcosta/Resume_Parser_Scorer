import streamlit as st
import re
import json
import google.generativeai as genai

genai.configure(api_key="api key here")  
model = genai.GenerativeModel("gemini-1.5-pro-latest")

def evaluate_code_logic(question_code, user_code):
    prompt = f"""
    You are a coding evaluator. 

    A candidate was asked to complete the following function:

    Question:
    {question_code}

    Their response:
    {user_code}

    Evaluate the candidate’s code **only for logical correctness**. Ignore variable names, spacing, and formatting.
    Return only one of the following JSON outputs:

    {{
      "correct": true,
      "feedback": "Brief explanation of why the logic is correct"
    }}

    OR

    {{
      "correct": false,
      "feedback": "Brief explanation of the mistake"
    }}
    """
    response = model.generate_content(prompt)
    match = re.search(r"\{.*\}", response.text, re.DOTALL)
    return json.loads(match.group(0)) if match else {"correct": False, "feedback": "Evaluation failed"}

st.set_page_config("Assessment", layout="wide")
st.title("🧪 Resume-Based Technical Assessment")

assessment = st.session_state.get("assessment_data")
if not assessment:
    st.warning("No assessment found. Please parse and score a resume first.")
    st.stop()

with st.form("assessment_form"):
    score = 0
    total = 0

    st.subheader("📘 Multiple Choice Questions")
    mcq_results = []
    for i, q in enumerate(assessment["mcqs"]):
        st.markdown(f"**Q{i+1}. {q['question']}**")
        ans = st.radio("", q["options"], key=f"mcq_{i}")
        mcq_results.append((ans, q["answer"]))
        st.markdown("---")

    st.subheader("✍️ Fill in the Blank")
    fib_results = []
    for i, q in enumerate(assessment["fill_in_the_blanks"]):
        st.markdown(f"**Q{i+1+len(mcq_results)}. {q['question']}**")
        ans = st.text_input("Answer:", key=f"fib_{i}")
        fib_results.append((ans.strip().lower(), q["answer"].strip().lower()))
        st.markdown("---")

    st.subheader("💻 Code Completion")
    code_results = []
    for i, q in enumerate(assessment["code_completion"]):
        st.markdown(f"**Q{i+1+len(mcq_results)+len(fib_results)}. Complete the code below:**")
        st.code(q["question"], language="python")
        ans = st.text_area("Your code:", key=f"code_{i}")
        code_results.append((ans.strip(), q["question"]))
        st.markdown("---")

    submit = st.form_submit_button("Submit Assessment")

if submit:
    st.subheader("📊 Results")

    for i, (given, correct) in enumerate(mcq_results):
        total += 1
        if given.strip().lower() == correct.strip().lower():
            st.success(f"✅ MCQ Q{i+1}: Correct")
            score += 1
        else:
            st.error(f"❌ MCQ Q{i+1}: Incorrect — Correct: {correct}")

    for i, (given, correct) in enumerate(fib_results):
        total += 1
        if given == correct:
            st.success(f"✅ Fill Q{i+1}: Correct")
            score += 1
        else:
            st.error(f"❌ Fill Q{i+1}: Incorrect — Correct: {correct}")

    for i, (user_code, question_code) in enumerate(code_results):
        total += 1
        if not user_code.strip():
            st.error(f"❌ Code Q{i+1}: No answer provided.")
            continue

        eval_result = evaluate_code_logic(question_code, user_code)
        if eval_result["correct"]:
            st.success(f"✅ Code Q{i+1}: Correct")
            st.caption(eval_result["feedback"])
            score += 1
        else:
            st.error(f"❌ Code Q{i+1}: Incorrect")
            st.caption(eval_result["feedback"])

    st.metric("🎓 Final Score", f"{score}/{total}")
    if score >= 0.7 * total:
        st.success("🎉 Well done!")
        st.balloons()
    else:
        st.info("Keep practicing and try again.")

