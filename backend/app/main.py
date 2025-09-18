import os
from fastapi import FastAPI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG Mistral Backend (Assignment #4)"}


import shutil
from fastapi import UploadFile, File
from typing import List
from app.ingestion import ingest_files, IngestedDoc

@app.post("/ingest/", response_model=List[IngestedDoc])
async def ingest(files: List[UploadFile] = File(...)):
    """
    Uploads and processes a list of PDF files for knowledge base ingestion.
    """
    # Create a temporary directory to store uploaded files
    temp_dir = "temp_docs"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_paths = []
    for file in files:
        # Sanitize filename to prevent directory traversal
        safe_filename = os.path.basename(file.filename)
        if not safe_filename or safe_filename.startswith('.'):
            raise ValueError(f"Invalid filename: {file.filename}")

        file_path = os.path.join(temp_dir, safe_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_paths.append(file_path)
        
    # Process the files
    ingestion_results = ingest_files(file_paths)
    
    # Clean up the temporary directory
    shutil.rmtree(temp_dir)
    
    return ingestion_results


from app.query import query_pipeline, QueryRequest, QueryResponse

@app.post("/query/", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Answers a user's question based on the ingested knowledge base.
    """
    return query_pipeline(request)


