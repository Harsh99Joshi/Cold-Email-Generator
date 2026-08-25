import json
import uuid

import chromadb

from config import COLLECTION_NAME, RESUME_PATH, VECTORSTORE_DIR


class Portfolio:
    def __init__(self, file_path=None):
        self.file_path = str(file_path or RESUME_PATH)
        self.data = self.load_json_data()
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        self.collection = self.chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    def load_json_data(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def project_document(project):
        tech = ", ".join(project.get("TechStack") or [])
        return (
            f"Project: {project.get('Name', '')}\n"
            f"Tech stack: {tech}\n"
            f"Description: {project.get('Description', '')}\n"
            f"Achievements: {project.get('Achievements', '')}"
        )

    def documents_by_name(self):
        return {
            project["Name"]: self.project_document(project)
            for project in self.data.get("AcademicProjects", [])
        }

    def load_portfolio(self, rebuild=False):
        if rebuild and self.collection.count():
            self.chroma_client.delete_collection(COLLECTION_NAME)
            self.collection = self.chroma_client.get_or_create_collection(name=COLLECTION_NAME)

        if self.collection.count():
            return

        ids, documents, metadatas = [], [], []
        for project in self.data.get("AcademicProjects", []):
            ids.append(str(uuid.uuid4()))
            documents.append(self.project_document(project))
            metadatas.append({"name": project["Name"]})

        if documents:
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def query_projects(self, skills, n_results=3, description=""):
        if isinstance(skills, str):
            query = skills
        else:
            query = ", ".join(str(skill) for skill in (skills or []) if skill)

        if description:
            query = f"{query}\n{description}".strip()

        if not query:
            return []

        available = self.collection.count()
        if not available:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, available),
            include=["documents", "metadatas", "distances"],
        )
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        projects = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}
            projects.append(
                {
                    "name": metadata.get("name", ""),
                    "document": document,
                    "distance": distance,
                }
            )
        return projects

    def query_links(self, skills):
        return ", ".join(project["name"] for project in self.query_projects(skills) if project["name"])

    def clear_collection(self):
        if self.collection.count():
            self.chroma_client.delete_collection(COLLECTION_NAME)
            self.collection = self.chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        print("ChromaDB collection cleared.")
