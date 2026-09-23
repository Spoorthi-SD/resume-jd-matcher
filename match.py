import re
import pdfplumber

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from domain_skills import SKILLS_BY_DOMAIN, DOMAIN_DESCRIPTIONS


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


def calculate_tfidf_score(resume_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words="english")

    tfidf_matrix = vectorizer.fit_transform(
        [resume_text, jd_text]
    )

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )

    return similarity[0][0]


def calculate_embedding_score(resume_text, jd_text, model):
    embeddings = model.encode(
        [resume_text, jd_text]
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )

    return similarity[0][0]


def detect_domain(jd_text, model, domain_descriptions):
    domains = list(domain_descriptions.keys())
    descriptions = list(domain_descriptions.values())

    jd_embedding = model.encode([jd_text])
    domain_embeddings = model.encode(descriptions)

    similarities = cosine_similarity(
        jd_embedding,
        domain_embeddings
    )[0]

    best_index = similarities.argmax()

    return (
        domains[best_index],
        round(float(similarities[best_index]) * 100, 2)
    )


def extract_skills(text, skill_list):
    found = []

    for skill in skill_list:
        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text):
            found.append(skill)

    return found


def find_missing_skills(resume_text, jd_text, skill_list):
    resume_skills = set(
        extract_skills(resume_text, skill_list)
    )

    jd_skills = set(
        extract_skills(jd_text, skill_list)
    )

    missing = jd_skills - resume_skills

    return (
        sorted(missing),
        sorted(jd_skills),
        sorted(resume_skills)
    )


if __name__ == "__main__":

    resume_text = clean_text(
        extract_text_from_pdf("sample_resume.pdf")
    )

    jd_raw_text = extract_text_from_txt("sample_jd.txt")
    jd_text = clean_text(jd_raw_text)

    # TF-IDF similarity
    tfidf_score = calculate_tfidf_score(
        resume_text,
        jd_text
    )

    # Load Sentence Transformer model
    print("Loading embedding model (first run may take a minute)...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Embedding similarity
    embedding_score = calculate_embedding_score(
        resume_text,
        jd_text,
        model
    )

    # Detect domain
    detected_domain, confidence = detect_domain(
        jd_text,
        model,
        DOMAIN_DESCRIPTIONS
    )

    print(
        f"\nDetected Domain: {detected_domain} "
        f"(confidence: {confidence}%)"
    )

    # Select skills based on detected domain
    active_skill_list = SKILLS_BY_DOMAIN[detected_domain]

    # Find skills
    missing, jd_skills, resume_skills = find_missing_skills(
        resume_text,
        jd_text,
        active_skill_list
    )

    # Display results
    print(
        f"\nTF-IDF Match Score: "
        f"{round(tfidf_score * 100, 2)}%"
    )

    print(
        f"Embedding Match Score: "
        f"{round(float(embedding_score) * 100, 2)}%"
    )

    print(
        f"\nSkills found in JD: {jd_skills}"
    )

    print(
        f"Skills found in Resume: {resume_skills}"
    )

    print(
        f"\nMissing Skills "
        f"(in JD but not in Resume): {missing}"
    )