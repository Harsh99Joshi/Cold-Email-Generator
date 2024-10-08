import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
from portfolio import Portfolio

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name='llama-3.1-70b-versatile'
        )
        self.portfolio = Portfolio()
        self.portfolio.load_portfolio()
        self.load_resume()

    def load_resume(self):
        with open("resource/resume.json", "r") as file:
            self.resume_data = json.load(file)

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### Scaped text from website:
            {page_data}
            ### Instruction:
            The scraped text is from the career page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )

        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too large. Unable to parse job")
        return res if isinstance(res, list) else [res]

    def get_relevant_projects(self, skills):
        relevant_projects = []
        for project in self.resume_data["AcademicProjects"]:
            # Check if any skill is mentioned in the project description or tech stack
            if any(skill in project["TechStack"] for skill in skills):
                relevant_projects.append(project["Name"])
        return relevant_projects

    def write_email(self, job, skills):
        # Query portfolio based on job skills
        job_description_str = str(job)  # Make sure job description is a string

        # Extract key details from the resume
        education = self.resume_data["Education"]
        experience = self.resume_data["ProfessionalExperience"]
        skills_list = self.resume_data["TechnicalSkills"]
        relevant_projects = self.get_relevant_projects(skills)

        # Build the email content conditionally
        education_section = f"**Education**: {', '.join([f'{edu['Degree']} from {edu['Institution']} ({edu['Dates']})' for edu in education])}\n" if education else ""
        experience_section = f"**Professional Experience**: {', '.join([f'{exp['Role']} at {exp['Company']} ({exp['Dates']})' for exp in experience])}\n" if experience else ""
        projects_section = f"**Relevant Projects**: {', '.join(relevant_projects)}\n" if relevant_projects else ""
        skills_section = f"**Skills**: {', '.join(skills_list['ProgrammingLanguages'])}\n" if skills_list else ""

        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Harshwardhan Joshi, a highly skilled Computer Science graduate with the following background:
            
            {education_section}
            {experience_section}
            {projects_section}
            {skills_section}
            
            Write a cold email applying for the job mentioned above, showcasing how your skills and relevant projects align with the job requirements.
            Oversell me as a candidate, and say only positive things. Make sure that they know I am ready to learn and grow.
            Do not add any information that has not been provided above. Do not make up any skills or projects.
            Do not provide a preamble. 
            
            ### EMAIL (NO PREAMBLE):
            """
        )

        # Pass the job description and resume details
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({
            "job_description": job_description_str,
            "education_section": education_section,
            "experience_section": experience_section,
            "projects_section": projects_section,
            "skills_section": skills_section,
        })
        return res.content
