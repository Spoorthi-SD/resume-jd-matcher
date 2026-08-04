import pdfplumber

def extract_text_from_pdf(filepath):
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def clean_text(text):
    text = text.lower()
    text = " ".join(text.split())
    return text

if __name__ == "__main__":
    resume_text = extract_text_from_pdf("sample_resume.pdf")
    cleaned = clean_text(resume_text)
    print(cleaned[:500])