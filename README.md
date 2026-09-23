\# Resume–JD Matcher



A machine learning-based application that compares a candidate's resume with a job description and provides a similarity score, detected job domain, and skill gap analysis.



\## Features



\- Upload a resume in PDF format

\- Upload a job description as PDF or TXT

\- Compare resume and job description using TF-IDF

\- Calculate semantic similarity using Sentence Transformers

\- Detect the likely job domain from the job description

\- Identify skills mentioned in the job description

\- Identify skills present in the resume

\- Show skills missing from the resume



\## Technologies Used



\- Python

\- FastAPI

\- Scikit-learn

\- Sentence Transformers

\- PDFPlumber

\- Uvicorn



\## Matching Methods



\### TF-IDF Similarity



TF-IDF is used to compare important words and terms in the resume and job description. Cosine similarity is then used to calculate the similarity score.



\### Sentence Transformer Embeddings



The application uses the `all-MiniLM-L6-v2` Sentence Transformer model to generate embeddings and calculate semantic similarity between the resume and job description.



\### Domain Detection



The job description is compared with predefined domain descriptions using sentence embeddings. The application identifies the domain with the highest similarity.



\### Skill Gap Analysis



The application checks domain-specific skills in both the resume and job description and identifies skills that are mentioned in the job description but missing from the resume.



\## Project Structure



```text

resume-jd-matcher/

│

├── main.py

├── match.py

├── domain\_skills.py

├── requirements.txt

├── extract\_text.py

└── .gitignore



\## Installation



Clone the repository:



git clone https://github.com/Spoorthi-SD/resume-jd-matcher.git



Go to the project folder:



cd resume-jd-matcher



Create a virtual environment:



python -m venv venv



Activate the virtual environment on Windows:



venv\\Scripts\\activate



Install the required packages:



pip install -r requirements.txt



\## Running the Backend



Start the FastAPI server:



uvicorn main:app --reload



The backend will run at:



http://localhost:8000



The main analysis endpoint is:



POST /api/analyze



\## Project Status



The backend is developed and tested locally. The React frontend is maintained in a separate repository.



\## Author



\*\*Spoorthi-SD\*\*

