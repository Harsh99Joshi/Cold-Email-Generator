"""Score how relevant retrieved portfolio projects are for sample jobs.

Uses RAGAS context precision / recall. LLM metrics need GROQ_API_KEY.
Non-LLM metrics and Hit@k always run.

Usage (from repo root, venv active):
    python "Cold Email Generator/app/evaluate_retrieval.py"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
os.chdir(APP_DIR)
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

import ragas_compat

ragas_compat.apply()

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    NonLLMContextPrecisionWithReference,
    NonLLMContextRecall,
)

from config import GROQ_API_KEY, GROQ_MODEL
from portfolio import Portfolio

TOP_K = 3
RESULTS_PATH = APP_DIR / "eval_results.csv"

EVAL_JOBS = [
    {
        "name": "LLM / RAG engineer",
        "skills": ["LLM", "LangChain", "RAG", "ChromaDB", "Streamlit", "Llama"],
        "description": (
            "Build retrieval-augmented LLM applications that extract job data, "
            "search a vector database, and generate personalized outreach emails."
        ),
        "expected_projects": ["Cold Email Generator"],
        "reference": (
            "The most relevant project is Cold Email Generator, which uses Llama, "
            "LangChain, ChromaDB, GROQ, and Streamlit to retrieve portfolio context "
            "and generate personalized cold emails."
        ),
    },
    {
        "name": "Deep learning time-series intern",
        "skills": ["Python", "LSTM", "GRU", "CNN", "TensorFlow", "time series"],
        "description": (
            "Forecast climate and sequential variables with deep learning sequence "
            "models such as LSTM, GRU, and CNN."
        ),
        "expected_projects": ["Multivariate Time Series Forecasting"],
        "reference": (
            "The most relevant project is Multivariate Time Series Forecasting, "
            "which implements LSTM, GRU, and CNN models in TensorFlow to predict "
            "climate variables from large-scale time series data."
        ),
    },
    {
        "name": "Big data recommendation engineer",
        "skills": ["Apache Spark", "MLlib", "PySpark", "SQL", "ALS", "collaborative filtering"],
        "description": (
            "Design a large-scale collaborative filtering recommender using Spark "
            "and MLlib on a dataset with more than a million interactions."
        ),
        "expected_projects": ["Book Recommendation System"],
        "reference": (
            "The most relevant project is Book Recommendation System, a Spark MLlib "
            "collaborative filtering pipeline using ALS on more than one million records."
        ),
    },
    {
        "name": "AWS ML engineer",
        "skills": ["AWS SageMaker", "S3", "EC2", "RDS", "Scikit-learn", "API"],
        "description": (
            "Deploy a scikit-learn classification model as a real-time API on AWS "
            "SageMaker with supporting S3, EC2, and RDS services."
        ),
        "expected_projects": ["Mobile Price Classification on AWS"],
        "reference": (
            "The most relevant project is Mobile Price Classification on AWS, which "
            "deploys a scikit-learn model as a SageMaker API with S3, EC2, and RDS."
        ),
    },
    {
        "name": "Java backend / SQL developer",
        "skills": ["Java", "SQL", "MSSQL", "JDBC", "database"],
        "description": (
            "Build a Java service that talks to Microsoft SQL Server through JDBC "
            "and writes reliable SQL for operational records."
        ),
        "expected_projects": ["Vehicle Servicing API"],
        "reference": (
            "The most relevant project is Vehicle Servicing API, a Java and MSSQL "
            "JDBC application that consolidates vehicle, customer, and mechanic records."
        ),
    },
]


def _hit_at_k(retrieved_names, expected_names, k):
    retrieved_top = retrieved_names[:k]
    return int(any(name in retrieved_top for name in expected_names))


def build_rows(portfolio: Portfolio):
    docs_by_name = portfolio.documents_by_name()
    rows = []
    retrieval_rows = []

    for case in EVAL_JOBS:
        retrieved = portfolio.query_projects(
            case["skills"],
            n_results=TOP_K,
            description=case["description"],
        )
        retrieved_names = [item["name"] for item in retrieved]
        retrieved_contexts = [item["document"] for item in retrieved]
        reference_contexts = [
            docs_by_name[name]
            for name in case["expected_projects"]
            if name in docs_by_name
        ]
        user_input = (
            f"Role: {case['name']}\n"
            f"Skills: {', '.join(case['skills'])}\n"
            f"Description: {case['description']}"
        )
        rows.append(
            {
                "user_input": user_input,
                "retrieved_contexts": retrieved_contexts,
                "reference_contexts": reference_contexts,
                "reference": case["reference"],
                "response": case["reference"],
            }
        )
        retrieval_rows.append(
            {
                "job": case["name"],
                "expected": ", ".join(case["expected_projects"]),
                "retrieved": " | ".join(retrieved_names) or "(none)",
                "hit@1": _hit_at_k(retrieved_names, case["expected_projects"], 1),
                "hit@3": _hit_at_k(retrieved_names, case["expected_projects"], 3),
                "precision@3": (
                    sum(name in case["expected_projects"] for name in retrieved_names[:TOP_K])
                    / max(len(retrieved_names[:TOP_K]), 1)
                ),
                "distances": ", ".join(
                    f"{item['name']}={float(item['distance']):.3f}" for item in retrieved
                ),
            }
        )
    return rows, retrieval_rows


def print_retrieval_table(retrieval_rows):
    print("\n=== Retrieval (Chroma) ===")
    for row in retrieval_rows:
        print(f"\nJob: {row['job']}")
        print(f"  Expected:  {row['expected']}")
        print(f"  Retrieved: {row['retrieved']}")
        print(f"  Distances: {row['distances']}")
        print(f"  Hit@1={row['hit@1']}  Hit@3={row['hit@3']}  Precision@3={row['precision@3']:.2f}")
    n = len(retrieval_rows)
    print(
        f"\nMean Hit@1={sum(r['hit@1'] for r in retrieval_rows) / n:.2f}  "
        f"Mean Hit@3={sum(r['hit@3'] for r in retrieval_rows) / n:.2f}  "
        f"Mean Precision@3={sum(r['precision@3'] for r in retrieval_rows) / n:.2f}"
    )


def run_ragas(rows, metrics, llm=None):
    dataset = EvaluationDataset.from_list(rows)
    kwargs = {"dataset": dataset, "metrics": metrics}
    if llm is not None:
        kwargs["llm"] = llm
    return evaluate(**kwargs)


def main():
    portfolio = Portfolio()
    print("Indexing projects into Chroma...")
    portfolio.load_portfolio(rebuild=True)
    print(f"Indexed {portfolio.collection.count()} projects.")

    rows, retrieval_rows = build_rows(portfolio)
    print_retrieval_table(retrieval_rows)

    print("\n=== RAGAS (non-LLM) ===")
    non_llm = run_ragas(
        rows,
        [
            NonLLMContextPrecisionWithReference(),
            NonLLMContextRecall(),
        ],
    )
    print(non_llm)
    result_df = non_llm.to_pandas()

    if GROQ_API_KEY:
        print(f"\n=== RAGAS (LLM judge: {GROQ_MODEL}) ===")
        from langchain_groq import ChatGroq

        evaluator_llm = LangchainLLMWrapper(
            ChatGroq(temperature=0, api_key=GROQ_API_KEY, model=GROQ_MODEL)
        )
        llm_result = run_ragas(
            rows,
            [
                LLMContextPrecisionWithReference(),
                LLMContextRecall(),
            ],
            llm=evaluator_llm,
        )
        print(llm_result)
        llm_df = llm_result.to_pandas()
        overlap = [col for col in llm_df.columns if col not in result_df.columns]
        if overlap:
            result_df = result_df.join(llm_df[overlap])
        else:
            result_df = llm_df
    else:
        print(
            "\nSkipping LLM RAGAS metrics (context precision / recall with a judge). "
            "Set GROQ_API_KEY in a .env file at the repo root to run them."
        )

    result_df.insert(0, "job", [row["job"] for row in retrieval_rows])
    result_df.insert(1, "expected", [row["expected"] for row in retrieval_rows])
    result_df.insert(2, "retrieved", [row["retrieved"] for row in retrieval_rows])
    result_df.insert(3, "hit@1", [row["hit@1"] for row in retrieval_rows])
    result_df.insert(4, "hit@3", [row["hit@3"] for row in retrieval_rows])
    result_df.insert(5, "precision@3", [row["precision@3"] for row in retrieval_rows])
    result_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
