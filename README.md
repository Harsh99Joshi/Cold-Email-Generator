# AI Cold Email Generator

An LLM-powered application that analyzes job postings and generates personalized outreach emails by retrieving the most relevant projects and experience from a candidate portfolio.

The application combines **web content extraction, retrieval-augmented generation (RAG), vector search, and Llama 3.1** to generate emails grounded in the requirements of a specific job rather than relying on generic templates.

## How It Works

1. The user provides a URL to a job posting.
2. The application extracts and processes the job description from the webpage.
3. Relevant information such as the role, required skills, and qualifications is identified.
4. Candidate projects and experience are stored and indexed in **ChromaDB**.
5. The system performs semantic retrieval to identify the projects most relevant to the job requirements.
6. The retrieved context and job description are provided to **Llama 3.1 through the Groq API**.
7. The model generates a personalized cold email highlighting the candidate experience most relevant to the role.
8. The generated email is displayed through a **Streamlit** interface.

## Architecture

```text
Job Posting URL
       |
       v
 Web Extraction
       |
       v
Job Description Processing
       |
       +----------------------+
       |                      |
       v                      v
 Role / Skill Extraction   Candidate Portfolio
                              |
                              v
                           ChromaDB
                              |
                              v
                       Semantic Retrieval
                              |
       +----------------------+
       |
       v
 Retrieved Relevant Projects
       |
       v
 Llama 3.1 via Groq
       |
       v
 Personalized Cold Email
       |
       v
 Streamlit Interface
```

## Key Features

### Retrieval-Augmented Generation

Candidate projects and experience are stored in a **ChromaDB vector database** and retrieved based on semantic similarity to the target job description. This allows the LLM to generate emails using relevant candidate experience instead of relying only on information contained in the prompt.

### Job Posting Analysis

The application extracts unstructured content from job posting webpages and converts it into structured information that can be used by the downstream retrieval and generation pipeline.

### Context-Aware Project Matching

Instead of including the same experience in every email, the system retrieves projects that most closely match the skills and requirements of the target position.

For example, an AI engineering job may retrieve projects involving LLMs and machine learning, while a backend role may retrieve projects involving APIs and databases.

### LLM-Based Email Generation

**Llama 3.1**, accessed through the **Groq API**, receives the job information and retrieved candidate context and generates a targeted outreach email.

### Interactive Interface

A **Streamlit** frontend allows users to provide job posting URLs and review the generated email through a lightweight web interface.

## Tech Stack

* **Language:** Python
* **LLM:** Llama 3.1
* **LLM Inference:** Groq API
* **Vector Database:** ChromaDB
* **AI Pattern:** Retrieval-Augmented Generation (RAG)
* **Frontend:** Streamlit
* **Web/Data Processing:** Python-based webpage extraction and parsing

## Running Locally

Create a virtual environment from the repo root and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `GROQ_API_KEY` from [Groq Console](https://console.groq.com/keys). Groq retired `llama-3.1-70b-versatile`; the default model is `openai/gpt-oss-120b`.

Launch the Streamlit app:

```powershell
.\.venv\Scripts\streamlit.exe run "Cold Email Generator/app/main.py"
```

Evaluate how relevant retrieved projects are with RAGAS (Hit@k plus context precision / recall):

```powershell
.\.venv\Scripts\python.exe "Cold Email Generator/app/evaluate_retrieval.py"
```

Non-LLM RAGAS metrics run without an API key. LLM-as-judge context precision and recall need `GROQ_API_KEY`. Results are written to `Cold Email Generator/app/eval_results.csv`.

## Example Workflow

**Input:** A job posting for an AI/ML Engineer requiring Python, LLM development, RAG, and database experience.

**Retrieval:** The application identifies candidate projects involving LLM applications, vector search, and machine learning.

**Output:** A personalized outreach email emphasizing the candidate's experience most relevant to those requirements.


## What I Learned

This project provided hands-on experience building an end-to-end LLM application combining **unstructured data extraction, semantic search, vector databases, retrieval-augmented generation, prompt engineering, and user-facing application development**.


# Screenshot
<img width="1225" alt="image" src="https://github.com/user-attachments/assets/6a35decc-dbda-434e-a37e-e112ecc88db4">
