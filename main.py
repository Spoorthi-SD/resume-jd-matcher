"""
FastAPI backend for the Resume-JD Matcher.

This file exposes the existing matching logic from match.py
through a REST API so the React frontend can use it.

SETUP:
1. Keep this file in the same folder as match.py and domain_skills.py.
2. Install required packages:
   pip install -r requirements.txt
3. Run with:
   uvicorn main:app --reload
4. API docs:
   http://localhost:8000/docs
"""

import tempfile
import os

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from match import (
    extract_text_from_pdf,
    extract_text_from_txt,
    clean_text,
    calculate_tfidf_score,
    calculate_embedding_score,
    detect_domain,
    find_missing_skills,
)

from domain_skills import SKILLS_BY_DOMAIN, DOMAIN_DESCRIPTIONS


app = FastAPI(title="Resume-JD Matcher API")


# Allow the React frontend to call this API
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


# Load the embedding model once when the server starts
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")


def _save_upload_to_temp(upload: UploadFile) -> str:
    """
    Save an uploaded file to a temporary file
    and return its file path.
    """

    suffix = os.path.splitext(upload.filename)[1]
    contents = upload.file.read()

    size_mb = len(contents) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            400,
            f"'{upload.filename}' is {size_mb:.1f}MB - "
            f"please upload a file under {MAX_FILE_SIZE_MB}MB."
        )

    if len(contents) == 0:
        raise HTTPException(
            400,
            f"'{upload.filename}' is empty."
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as tmp:
        tmp.write(contents)
        return tmp.name


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Resume-JD Matcher API is running"
    }


@app.post("/api/analyze")
async def analyze(
    resume: UploadFile = File(
        ...,
        description="Resume PDF"
    ),

    jd_file: UploadFile | None = File(
        None,
        description="Job description file (.pdf or .txt)"
    ),

    jd_text: str | None = Form(
        None,
        description="Job description as raw pasted text"
    ),
):
    """
    Accepts a resume PDF and a job description.

    The job description can be provided as:
    - PDF file
    - TXT file
    - Pasted text

    Returns:
    - Detected job domain
    - Domain confidence
    - TF-IDF similarity score
    - Embedding similarity score
    - Skills found in JD
    - Skills found in resume
    - Missing skills
    """

    # Validate resume
    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(
            400,
            "Resume must be a .pdf file."
        )


    # Check whether JD file or JD text was provided
    has_jd_file = (
        jd_file is not None
        and jd_file.filename
    )

    has_jd_text = bool(
        jd_text and jd_text.strip()
    )


    if not has_jd_file and not has_jd_text:
        raise HTTPException(
            400,
            "Provide either a job description file or pasted JD text."
        )


    # Validate JD file extension
    if (
        has_jd_file
        and not jd_file.filename.lower().endswith(
            ALLOWED_JD_EXTENSIONS
        )
    ):
        raise HTTPException(
            400,
            "Job description file must be .pdf or .txt."
        )


    resume_path = None
    jd_path = None


    try:

        # -------------------------------------------------
        # Resume processing
        # -------------------------------------------------

        resume_path = _save_upload_to_temp(resume)

        resume_text = clean_text(
            extract_text_from_pdf(resume_path)
        )


        if not resume_text.strip():
            raise HTTPException(
                422,
                "Couldn't extract any text from that resume PDF. "
                "It may be a scanned image rather than a text-based PDF."
            )


        # -------------------------------------------------
        # Job description processing
        # -------------------------------------------------

        if has_jd_file:

            jd_path = _save_upload_to_temp(jd_file)

            if jd_file.filename.lower().endswith(".pdf"):

                jd_raw_text = extract_text_from_pdf(
                    jd_path
                )

            else:

                jd_raw_text = extract_text_from_txt(
                    jd_path
                )

        else:

            jd_raw_text = jd_text


        jd_text_clean = clean_text(
            jd_raw_text
        )


        if not jd_text_clean.strip():
            raise HTTPException(
                422,
                "Couldn't extract any text from the job description provided."
            )


        # -------------------------------------------------
        # Matching
        # -------------------------------------------------

        # TF-IDF similarity
        tfidf_score = calculate_tfidf_score(
            resume_text,
            jd_text_clean
        )


        # Sentence Transformer embedding similarity
        embedding_score = calculate_embedding_score(
            resume_text,
            jd_text_clean,
            model
        )


        # Detect job domain
        detected_domain, confidence = detect_domain(
            jd_text_clean,
            model,
            DOMAIN_DESCRIPTIONS
        )


        # Select skills based on detected domain
        active_skill_list = SKILLS_BY_DOMAIN[
            detected_domain
        ]


        # Find skills and missing skills
        missing, jd_skills, resume_skills = find_missing_skills(
            resume_text,
            jd_text_clean,
            active_skill_list
        )


        # -------------------------------------------------
        # Return results
        # -------------------------------------------------

        return {

            "domain": {
                "detected": detected_domain,
                "confidence_percent": confidence,
            },

            "scores": {
                "tfidf_percent": round(
                    tfidf_score * 100,
                    2
                ),

                "embedding_percent": round(
                    float(embedding_score) * 100,
                    2
                ),
            },

            "skills": {
                "in_jd": jd_skills,

                "in_resume": resume_skills,

                "missing_from_resume": missing,
            },
        }


    except HTTPException:
        # Pass intentional HTTP errors through
        raise


    except Exception as exc:

        print(
            f"Unexpected error during analysis: {exc}"
        )

        raise HTTPException(
            500,
            "Something went wrong while analyzing these files. "
            "Please try again."
        )


    finally:

        # Delete temporary files
        for path in (
            resume_path,
            jd_path
        ):

            if path and os.path.exists(path):
                os.remove(path)