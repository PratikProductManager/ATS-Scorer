import os
import uuid
import streamlit as st
from RAG_Utility import process_documents_to_chroma, evaluate_resume

# -------------------------------------------------
# Setup
# -------------------------------------------------
working_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="ATS Resume Scorer- Powered by Pratik", page_icon="📄")

st.title("📄 ATS Resume Scorer")

# -------------------------------------------------
# Upload Resume
# -------------------------------------------------
uploaded_resume = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

# Job description
job_description = st.text_area("Paste Job Description")

# Session state to prevent reprocessing
if "processed" not in st.session_state:
    st.session_state.processed = False

# -------------------------------------------------
# Process Resume
# -------------------------------------------------
if uploaded_resume is not None and not st.session_state.processed:

    unique_name = str(uuid.uuid4()) + ".pdf"
    save_path = os.path.join(working_dir, unique_name)

    with open(save_path, "wb") as f:
        f.write(uploaded_resume.getbuffer())

    with st.spinner("Processing resume and ATS guidelines..."):
        process_documents_to_chroma(unique_name)

    st.session_state.processed = True
    st.success("Resume processed successfully ✔")

# -------------------------------------------------
# Evaluate ATS Score
# -------------------------------------------------
if st.button("Calculate ATS Score"):

    if uploaded_resume is None:
        st.warning("Please upload your resume")

    elif job_description.strip() == "":
        st.warning("Please paste the job description")

    else:
        with st.spinner("Evaluating resume with Llama-3.3-70B..."):
            result = evaluate_resume(job_description)

        st.markdown("## 📊 ATS Evaluation Result")
        st.markdown(result)