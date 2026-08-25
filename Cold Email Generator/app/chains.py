import json

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from config import GROQ_MODEL, RESUME_PATH, require_groq_key
from portfolio import Portfolio


class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            api_key=require_groq_key(),
            model=GROQ_MODEL,
        )
        self.portfolio = Portfolio()
        self.portfolio.load_portfolio()
        self.load_resume()

    def load_resume(self):
        with open(RESUME_PATH, "r", encoding="utf-8") as file:
            self.resume_data = json.load(file)

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### Scraped text from website:
            {page_data}
            ### Instruction:
            The scraped text is from the career page of a website.
            Extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            parsed = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too large. Unable to parse job")
        return parsed if isinstance(parsed, list) else [parsed]

    def write_email(self, job, retrieved_projects):
        job_description_str = str(job)
        education = self.resume_data.get("Education") or []
        experience = self.resume_data.get("ProfessionalExperience") or []
        skills_list = self.resume_data.get("TechnicalSkills") or {}

        education_section = ", ".join(
            f"{edu.get('Degree')} from {edu.get('Institution')} ({edu.get('Dates')})"
            for edu in education
        )
        experience_section = ", ".join(
            f"{exp.get('Role')} at {exp.get('Company')} ({exp.get('Dates')})"
            for exp in experience
        )
        skills_section = ", ".join(skills_list.get("ProgrammingLanguages") or [])

        if isinstance(retrieved_projects, str):
            projects_section = retrieved_projects
        else:
            project_lines = []
            for project in retrieved_projects or []:
                if isinstance(project, dict):
                    name = project.get("name") or ""
                    document = project.get("document") or ""
                    project_lines.append(f"{name}: {document}".strip(": "))
                else:
                    project_lines.append(str(project))
            projects_section = "\n".join(project_lines)

        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Harshwardhan Joshi, a highly skilled Computer Science graduate with the following background:

            Education: {education_section}
            Professional Experience: {experience_section}
            Relevant retrieved projects:
            {projects_section}
            Skills: {skills_section}

            Write a cold email applying for the job mentioned above, showcasing how your skills and relevant projects align with the job requirements.
            Oversell me as a candidate, and say only positive things. Make sure that they know I am ready to learn and grow.
            Do not add any information that has not been provided above. Do not make up any skills or projects.
            Do not provide a preamble.

            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke(
            {
                "job_description": job_description_str,
                "education_section": education_section,
                "experience_section": experience_section,
                "projects_section": projects_section,
                "skills_section": skills_section,
            }
        )
        return res.content
