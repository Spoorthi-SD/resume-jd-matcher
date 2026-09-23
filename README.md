# Resume–JD Matcher (Backend)

A machine learning-based application that compares a candidate's resume with a job description and provides a similarity score, detected job domain, and skill gap analysis.

> Frontend repo: [resume-jd-matcher-frontend](https://github.com/Spoorthi-SD/resume-jd-matcher-frontend)

## Features

- Upload a resume in PDF format
- Upload a job description as PDF or TXT
- Compare resume and job description using TF-IDF
- Calculate semantic similarity using Sentence Transformers
- Detect the likely job domain from the job description
- Identify skills mentioned in the job description
- Identify skills present in the resume
- Show skills missing from the resume

## Technologies Used

- Python
- FastAPI
- Scikit-learn
- Sentence Transformers
- PDFPlumber
- Uvicorn

## Matching Methods

### TF-IDF Similarity
TF-IDF is used to compare important words and terms in the resume and job description. Cosine similarity is then used to calculate the similarity score.

### Sentence Transformer Embeddings
The application uses the `all-MiniLM-L6-v2` Sentence Transformer model to generate embeddings and calculate semantic similarity between the resume and job description.

### Domain Detection
The job description is compared with predefined domain descriptions using sentence embeddings. The application identifies the domain with the highest similarity.

### Skill Gap Analysis
The application checks domain-specific skills in both the resume and job description and identifies skills that are mentioned in the job description but missing from the resume.

## Project Structure

```
resume-jd-matcher/
│
├── main.py
├── match.py
├── domain_skills.py
├── extract_text.py
├── requirements.txt
└── .gitignore
```

## Installation

Clone the repository:
```bash
git clone https://github.com/Spoorthi-SD/resume-jd-matcher.git
cd resume-jd-matcher
```

Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Backend

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The backend will run at:
```
http://localhost:8000
```

Interactive API docs (Swagger UI) are auto-generated at:
```
http://localhost:8000/docs
```

## API Reference

### `GET /`

Health check. Returns `{"status": "ok", "message": "Resume-JD Matcher API is running"}`.

### `POST /api/analyze`

Analyzes a resume against a job description. The job description can be provided either as a file **or** as pasted text.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|---|---|---|---|
| `resume` | file (PDF) | Yes | Candidate's resume |
| `jd_file` | file (PDF/TXT) | One of `jd_file` or `jd_text` | Job description as a file |
| `jd_text` | text | One of `jd_file` or `jd_text` | Job description as pasted text |

**Response:**
```json
{
  "domain": {
    "detected": "Software Engineering",
    "confidence_percent": 81.4
  },
  "scores": {
    "tfidf_percent": 61.7,
    "embedding_percent": 78.3
  },
  "skills": {
    "in_jd": ["python", "sql", "rest api", "docker"],
    "in_resume": ["python", "sql", "rest api"],
    "missing_from_resume": ["docker"]
  }
}
```

**Notes:**
- Max upload size is 10MB per file.
- The resume must be a text-based PDF (scanned image PDFs won't extract text).
- By default, CORS is only enabled for `http://localhost:5173` (the Vite dev server). Add your deployed frontend URL to `allow_origins` in `main.py` before hosting.

## Project Status

The backend is developed and tested locally. The React frontend is maintained in a separate repository (linked above).

## Author

**Spoorthi-SD**