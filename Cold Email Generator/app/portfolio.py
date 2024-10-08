import pandas as pd
import json
import chromadb
import uuid


class Portfolio:
    def __init__(self, file_path="resource/resume.json"):
        self.file_path = file_path
        self.data = self.load_json_data()
        self.chroma_client = chromadb.PersistentClient('vectorstore')
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_json_data(self):
        # Load the resume JSON file
        with open(self.file_path, "r") as file:
            data = json.load(file)
        return data

    def load_portfolio(self):
        if not self.collection.count():
            # Add technical skills and projects to the collection
            for skill in self.data['TechnicalSkills']:
                self.collection.add(
                    documents=str(skill),
                    metadatas={"links": "Resume Link"},  # Add link to resume if needed
                    ids=[str(uuid.uuid4())]
                )
            for project in self.data['AcademicProjects']:
                self.collection.add(
                    documents=project['Description'],
                    metadatas={"links": project['Name']},
                    ids=[str(uuid.uuid4())]
                )

    def query_links(self, skills):
        # Query relevant projects or skills based on the job description's skills
        results = self.collection.query(query_texts=skills, n_results=1)
        
        # Get the 'metadatas' list from the query result
        metadata_list = results.get('metadatas', [])
        
        # Ensure we're iterating over the list correctly and accessing elements
        links = []
        for metadata in metadata_list:
            if isinstance(metadata, list):  # Handle if 'metadatas' is a list of lists
                for entry in metadata:
                    if 'links' in entry:
                        links.append(entry['links'])
            elif 'links' in metadata:  # If it's a flat list of dictionaries
                links.append(metadata['links'])

        # Convert the links into a comma-separated string
        return ', '.join(links)
    
    def clear_collection(self):
        """
        Clears all documents from the ChromaDB collection.
        """
        self.collection.delete()  # Deletes all documents in the collection
        print("ChromaDB collection cleared.")
