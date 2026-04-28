import streamlit as st
import pandas as pd
import re
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------------- PDF TO TEXT ----------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


# ---------------- TF-IDF KEYWORDS ----------------
def get_keywords(vectorizer, tfidf_matrix):
    feature_names = vectorizer.get_feature_names_out()
    job_vector = tfidf_matrix[0].toarray()[0]

    words = []

    for i, score in enumerate(job_vector):
        word = feature_names[i]

        if (
            word not in ENGLISH_STOP_WORDS
            and len(word) > 2
            and word.isalpha()
        ):
            words.append((word, score))

    words.sort(key=lambda x: x[1], reverse=True)

    return [w[0] for w in words[:7]]


# ---------------- SKILL MATCH SCORE ----------------
def skill_match_score(job_text, resume_text):
    job_words = set(job_text.split())
    resume_words = set(resume_text.split())

    common = job_words.intersection(resume_words)

    if len(job_words) == 0:
        return 0

    return round((len(common) / len(job_words)) * 100, 2)


# ---------------- MISSING SKILLS ----------------
def missing_skills(job_text, resume_text):
    job_words = set(job_text.split())
    resume_words = set(resume_text.split())

    missing = job_words - resume_words
    return list(missing)[:10]


# ---------------- UI ----------------
st.title("📄 AI Resume Ranking System - Portfolio Level")
st.write("Upload resumes and match them with job description")

job_desc = st.text_area("Enter Job Description")

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF only)",
    type=["pdf"],
    accept_multiple_files=True
)


if st.button("Rank Resumes"):

    if not job_desc or not uploaded_files:
        st.warning("Please add job description and upload resumes")
    else:

        resumes = []
        names = []

        # Extract resumes
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            resumes.append(text)
            names.append(file.name)

        # Clean
        job_clean = clean_text(job_desc)
        resumes_clean = [clean_text(r) for r in resumes]

        # TF-IDF
        docs = [job_clean] + resumes_clean
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(docs)

        # Similarity
        scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

        # Results table
        df = pd.DataFrame({
            "Resume": names,
            "Match Score (%)": (scores * 100).round(2)
        }).sort_values(by="Match Score (%)", ascending=False)

        st.subheader("📊 Ranking Results")
        st.dataframe(df)

        # Top skills
        st.subheader("🔥 Top Skills in Job Description")
        st.write(get_keywords(vectorizer, tfidf))

        # Detailed analysis
        st.subheader("🧠 Resume Analysis")

        for i in range(len(resumes_clean)):
            st.write(f"📄 {names[i]}")
            st.write(f"Skill Match Score: {skill_match_score(job_clean, resumes_clean[i])}%")
            st.write(f"Missing Skills: {missing_skills(job_clean, resumes_clean[i])}")
            st.write("---")

        st.success("Analysis Complete ✔")