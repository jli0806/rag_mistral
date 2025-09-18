import os
from typing import List, Dict, Any
from pydantic import BaseModel
from pypdf import PdfReader
from mistralai import Mistral
from app.store import store

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if MISTRAL_API_KEY is None:
    raise ValueError("MISTRAL_API_KEY environment variable is not set.")

client = Mistral(api_key=MISTRAL_API_KEY)

class IngestedDoc(BaseModel):
    file_name: str
    num_chunks: int


def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF {file_path}: {str(e)}")

def chunk_text(text: str, chunk_size: int = 1024, chunk_overlap: int = 200) -> List[str]:
    """
    Splits text into smaller chunks with overlap to maintain context.

    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk in characters (default: 1024)
        chunk_overlap: Overlap between chunks to preserve context (default: 200)

    Returns:
        List of text chunks

    Chunking considerations:
    - Chunk size balances granular context vs broader meaning
    - Overlap prevents information loss at boundaries
    - Character-based splitting for simplicity
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a list of texts using the Mistral API."""
    if not texts:
        return []

    try:
        # Process in batches to avoid API limits
        batch_size = 100  # Mistral API batch limit
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings_response = client.embeddings.create(
                model="mistral-embed",
                inputs=batch
            )
            batch_embeddings = [embedding.embedding for embedding in embeddings_response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings
    except Exception as e:
        raise ValueError(f"Failed to generate embeddings: {str(e)}")

def ingest_files(file_paths: List[str]) -> List[IngestedDoc]:
    """
    Main ingestion pipeline: extract text -> chunk -> generate embeddings -> store.
    """
    processed_files = []
    
    for file_path in file_paths:
        try:
            # 1. Extract text
            text = extract_text_from_pdf(file_path)
            
            # 2. Chunk text
            text_chunks = chunk_text(text)
            
            if not text_chunks:
                processed_files.append(IngestedDoc(file_name=os.path.basename(file_path), num_chunks=0))
                continue

            # 3. Generate embeddings
            embeddings = get_embeddings(text_chunks)
            
            # 4. Prepare for storage
            chunks_to_store = [
                {
                    "file_name": os.path.basename(file_path),
                    "chunk_text": chunk,
                    "embedding": emb
                }
                for chunk, emb in zip(text_chunks, embeddings)
            ]
            
            # 5. Store chunks
            store.add_chunks(chunks_to_store)
            
            processed_files.append(IngestedDoc(file_name=os.path.basename(file_path), num_chunks=len(text_chunks)))

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            processed_files.append(IngestedDoc(file_name=os.path.basename(file_path), num_chunks=-1))

    return processed_files