# Resume Review Agent

A beginner-friendly **single-agent CrewAI application** that compares a candidate resume with a target job description.

The app uses:

- **Streamlit** for the user interface
- **CrewAI** for the agent/task/crew workflow
- **Groq** as the LLM provider
- **pypdf** for local PDF text extraction

## Architecture

The project intentionally uses:

- exactly **1 CrewAI Agent**
- exactly **1 CrewAI Task**
- exactly **1 Crew**
- no database
- no vector database
- no RAG
- no authentication
- no Docker

The app accepts either pasted resume text or an uploaded text-based PDF, plus a pasted job description.

The reviewer produces these sections:

1. Match Summary
2. Skills Found
3. Missing Requirements
4. Unclear / Not Demonstrated
5. Experience Gaps
6. Education / Qualification Gaps
7. Resume Improvements
8. Keywords to Consider
9. Priority Action Plan

## Accuracy Rule

The prompt explicitly tells the agent not to invent or assume qualifications, experience, projects, certifications, skills, achievements, employment history, or education.

When the evidence is insufficient, the app tells the model to classify the requirement as **Unknown / Not Demonstrated**.

## Project Structure

```text
resume-review-agent/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml.example
```

## Requirements

- Python **3.11**
- A Groq API key

Pinned main dependencies:

```text
streamlit==1.32.2
crewai==0.80.0
pypdf==4.1.0
```

## Local Setup

### 1. Create a virtual environment

Windows:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create your local Streamlit secrets file

Copy:

```text
.streamlit/secrets.toml.example
```

to:

```text
.streamlit/secrets.toml
```

Then replace the placeholder key:

```toml
GROQ_API_KEY = "gsk_your_real_groq_api_key_here"
GROQ_MODEL = "openai/gpt-oss-120b"
```

Do **not** commit `.streamlit/secrets.toml`.

### 4. Run the app

```bash
streamlit run app.py
```

## GitHub Setup

1. Create a new GitHub repository, for example `resume-review-agent`.
2. Upload:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `.gitignore`
   - `.streamlit/secrets.toml.example`
3. Do **not** upload a real `.streamlit/secrets.toml`.
4. Commit the files.

## Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud.
2. Choose **Create app** / **New app**.
3. Select your GitHub repository.
4. Set the main file path to:
   ```text
   app.py
   ```
5. In Advanced settings, select **Python 3.11**.
6. Open the app's **Secrets** settings and add:

```toml
GROQ_API_KEY = "gsk_your_real_groq_api_key_here"
GROQ_MODEL = "openai/gpt-oss-120b"
```

7. Deploy the app.

Never put your real API key in `app.py`, `README.md`, or any public GitHub file.

## PDF Notes

`pypdf` extracts selectable text from normal PDFs.

This minimal application intentionally does **not** include OCR. A scanned/image-only resume may contain no extractable text. In that case:

- paste the resume text manually, or
- convert the PDF into a searchable/text-based PDF first.

## Common Errors

### `GROQ_API_KEY is missing`

Add the key to:

- local `.streamlit/secrets.toml`, or
- Streamlit Community Cloud → App settings → Secrets.

### Authentication / 401 error

The API key may be invalid, expired, revoked, or copied incorrectly. Create/check the key in your Groq account and update Streamlit Secrets.

### Rate limit / 429 error

The Groq account has reached a request/token limit. Wait and retry, shorten very large inputs, or review the limits for your Groq account.

### Model unavailable / 404 error

Check `GROQ_MODEL`. This project defaults to:

```text
openai/gpt-oss-120b
```

The app automatically adds CrewAI's Groq provider prefix internally.

### PDF has no text

The PDF may be scanned or image-only. Paste the resume text or convert the file to a searchable PDF.

### Dependency installation problem

Confirm Streamlit Cloud is using **Python 3.11** and that `requirements.txt` is present at the repository root.

## Privacy

The project does not intentionally save resume or job-description content in a database or persistent project file.

Uploaded PDF text is extracted locally in memory. The resume/job-description text is then sent to the configured Groq LLM service for analysis.

For real HR or production use, review your organization's privacy, consent, retention, and data-processing requirements before sending personal information to any third-party AI provider.
