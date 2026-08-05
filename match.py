import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(filepath):
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def clean_text(text):
    text = text.lower()
    text = " ".join(text.split())
    return text

def calculate_match_score(resume_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]

if __name__ == "__main__":
    resume_text = clean_text(extract_text_from_pdf("sample_resume.pdf"))
    jd_text = clean_text(extract_text_from_txt("sample_jd.txt"))

    score = calculate_match_score(resume_text, jd_text)
    match_percentage = round(score * 100, 2)

    print(f"Match Score: {match_percentage}%")