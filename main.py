"""
FastAPI backend for the Resume-JD Matcher.

This file does NOT reimplement your matching logic - it imports the
functions you already wrote in match.py and domain_skills.py, and just
exposes them over HTTP so a frontend can call them.

SETUP:
1. Copy this file into your existing resume-jd-matcher project folder
   (same folder as match.py and domain_skills.py).
2. pip install fastapi uvicorn python-multipart
3. Run with:  uvicorn main:app --reload
4. API docs auto-generated at http://localhost:8000/docs
"""

import io
import tempfile
import os

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

# --- Import your existing logic, unchanged ---
from match import (
    extract_text_from_pdf,
    extract_text_from_txt,
    clean_text,
    calculate_tfidf_score,
    calculate_embedding_score,
    detect_domain,
    find_missing_skills,
    extract_noun_phrases,
)
from domain_skills import SKILLS_BY_DOMAIN, DOMAIN_DESCRIPTIONS

app = FastAPI(title="Resume-JD Matcher API")

# Allow the React dev server to call this API. Listed explicitly (rather
# than "*") so the browser only trusts requests from your own frontend -
# add your deployed frontend URL here later when you host it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE_MB = 10
ALLOWED_JD_EXTENSIONS = (".pdf", ".txt")

# Load the embedding model once at startup, not on every request -
# this is the slow step (a few seconds), so we pay that cost once.
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")


def _save_upload_to_temp(upload: UploadFile) -> str:
    """Writes an uploaded file to a temp path and returns that path,
    since pdfplumber/pdf readers expect a filepath, not raw bytes."""
    suffix = os.path.splitext(upload.filename)[1]
    contents = upload.file.read()

    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            400, f"'{upload.filename}' is {size_mb:.1f}MB - please upload a file under {MAX_FILE_SIZE_MB}MB."
        )
    if len(contents) == 0:
        raise HTTPException(400, f"'{upload.filename}' is empty.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        return tmp.name


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Resume-JD Matcher API is running"}


@app.post("/api/analyze")
async def analyze(
    resume: UploadFile = File(..., description="Resume PDF"),
    jd_file: UploadFile | None = File(None, description="Job description file (.pdf or .txt)"),
    jd_text: str | None = Form(None, description="Job description as raw pasted text"),
):
    """
    Accepts a resume PDF plus a job description - either as an uploaded
    file (.pdf or .txt) OR as pasted text - and returns the full match
    analysis as JSON.
    """
    # --- Validate inputs up front, before doing any work ---
    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Resume must be a .pdf file.")

    has_jd_file = jd_file is not None and jd_file.filename
    has_jd_text = bool(jd_text and jd_text.strip())

    if not has_jd_file and not has_jd_text:
        raise HTTPException(400, "Provide either a job description file or pasted JD text.")

    if has_jd_file and not jd_file.filename.lower().endswith(ALLOWED_JD_EXTENSIONS):
        raise HTTPException(400, "Job description file must be .pdf or .txt.")

    resume_path = None
    jd_path = None
    try:
        # --- Resume ---
        resume_path = _save_upload_to_temp(resume)
        resume_text = clean_text(extract_text_from_pdf(resume_path))

        if not resume_text.strip():
            raise HTTPException(
                422,
                "Couldn't extract any text from that resume PDF. "
                "It may be a scanned image rather than a text-based PDF.",
            )

        # --- Job description ---
        if has_jd_file:
            jd_path = _save_upload_to_temp(jd_file)
            if jd_file.filename.lower().endswith(".pdf"):
                jd_raw_text = extract_text_from_pdf(jd_path)
            else:
                jd_raw_text = extract_text_from_txt(jd_path)
        else:
            jd_raw_text = jd_text

        jd_text_clean = clean_text(jd_raw_text)

        if not jd_text_clean.strip():
            raise HTTPException(422, "Couldn't extract any text from the job description provided.")

        # --- Run your existing pipeline ---
        tfidf_score = calculate_tfidf_score(resume_text, jd_text_clean)
        embedding_score = calculate_embedding_score(resume_text, jd_text_clean, model)
        detected_domain, confidence = detect_domain(jd_text_clean, model, DOMAIN_DESCRIPTIONS)

        active_skill_list = SKILLS_BY_DOMAIN[detected_domain]
        missing, jd_skills, resume_skills = find_missing_skills(
            resume_text, jd_text_clean, active_skill_list
        )
        candidate_phrases = extract_noun_phrases(jd_raw_text, set(active_skill_list))

        return {
            "domain": {
                "detected": detected_domain,
                "confidence_percent": confidence,
            },
            "scores": {
                "tfidf_percent": round(tfidf_score * 100, 2),
                "embedding_percent": round(float(embedding_score) * 100, 2),
            },
            "skills": {
                "in_jd": jd_skills,
                "in_resume": resume_skills,
                "missing_from_resume": missing,
            },
            "candidate_phrases": candidate_phrases,
        }

    except HTTPException:
        raise  # already a clean, intentional error - pass it through as-is
    except Exception as exc:
        # Anything unexpected (corrupt PDF, model hiccup, etc.) - log the
        # real error server-side, but don't leak a raw stack trace to the client
        print(f"Unexpected error during analysis: {exc}")
        raise HTTPException(500, "Something went wrong while analyzing these files. Please try again.")
    finally:
        # Clean up temp files regardless of success/failure
        for path in (resume_path, jd_path):
            if path and os.path.exists(path):
                os.remove(path)

