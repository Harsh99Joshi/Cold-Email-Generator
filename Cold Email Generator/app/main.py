import streamlit as st

from chains import Chain
from portfolio import Portfolio
from utils import clean_text, load_job_text


def create_streamlit_app(llm, portfolio, clean_text_fn):
    st.title("📧 Cold Mail Generator")
    url_input = st.text_input("Enter a URL:", value="")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            data = clean_text_fn(load_job_text(url_input))
            portfolio.load_portfolio()
            jobs = llm.extract_jobs(data)
            for job in jobs:
                skills = job.get("skills", [])
                projects = portfolio.query_projects(
                    skills,
                    n_results=3,
                    description=job.get("description", ""),
                )
                st.subheader(job.get("role", "Retrieved projects"))
                if projects:
                    st.dataframe(
                        [
                            {
                                "project": project["name"],
                                "distance": round(float(project["distance"]), 4),
                            }
                            for project in projects
                        ],
                        hide_index=True,
                    )
                else:
                    st.info("No projects were retrieved for this job.")
                email = llm.write_email(job, projects)
                st.code(email, language="markdown")
        except Exception as e:
            st.error(f"An Error Occurred: {e}")


if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, portfolio, clean_text)
