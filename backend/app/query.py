import os
import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Any
from mistralai import Mistral

from app.store import store
from app.ingestion import get_embeddings

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if MISTRAL_API_KEY is None:
    raise ValueError("MISTRAL_API_KEY environment variable is not set.")

client = Mistral(api_key=MISTRAL_API_KEY)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: List[Dict[str, Any]]

# --- Core Functions ---

def is_greeting(query: str) -> bool:
    """Simple intent detection for greetings."""
    greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    return query.lower().strip() in greetings

def transform_query(query: str) -> str:
    """
    Transforms user query for improved retrieval by rephrasing for better semantic matching.
    """
    try:
        messages = [
            {"role": "system", "content": "You are a query optimization assistant. Your task is to rephrase the user's question into a more effective query for a semantic search engine. Focus on key terms and concepts."},
            {"role": "user", "content": f"Original query: {query}"}
        ]
        
        chat_response = client.chat.complete(
            model="mistral-small-latest",
            messages=messages,
            temperature=0.1
        )
        
        transformed_query = chat_response.choices[0].message.content
        return transformed_query.strip()
    except Exception as e:
        print(f"Error transforming query: {e}")
        return query # Fallback to the original query

def semantic_search(query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
    """
    Performs semantic search using cosine similarity to find relevant chunks.

    Args:
        query_embedding: Vector representation of the query
        top_k: Number of top results to return

    Returns:
        List of most relevant chunks with metadata

    Note: Hybrid search could combine semantic and keyword approaches:
    - TF-IDF/BM25 for keyword relevance
    - Cosine similarity for semantic relevance
    - Weighted combination of both scores
    """
    try:
        all_chunks = store.get_all_chunks()
        if not all_chunks:
            return []

        embeddings = np.array([chunk["embedding"] for chunk in all_chunks])
        query_embedding = np.array(query_embedding)

        # Calculate cosine similarity
        similarities = np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding))

        # Get top_k indices
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]

        return [all_chunks[i] for i in top_k_indices]
    except Exception as e:
        print(f"Error in semantic search: {e}")
        return []

def generate_answer(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Generates an answer using the LLM based on the retrieved context.
    """
    context = "\n\n---\n\n".join([chunk["chunk_text"] for chunk in retrieved_chunks])
    
    prompt = f"""
    You are a helpful assistant. Based on the following context, please answer the user's question.

    **Context:**
    {context}

    **Question:**
    {query}

    **Answer:**
    """

    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ]
        
        chat_response = client.chat.complete(
            model="mistral-small-latest",
            messages=messages,
            temperature=0.7
        )
        
        return chat_response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "I'm sorry, I couldn't generate an answer at this time."

def query_pipeline(request: QueryRequest) -> QueryResponse:
    """
    Main query processing pipeline: intent detection -> search -> answer generation.
    """
    # 1. Intent Detection
    if is_greeting(request.query):
        return QueryResponse(answer="Hello! How can I help you today?", retrieved_chunks=[])

    # 2. Query Transformation (optional, can be enabled for more complex scenarios)
    # transformed_query = transform_query(request.query)
    transformed_query = request.query 

    # 3. Get query embedding
    query_embedding = get_embeddings([transformed_query])[0]

    # 4. Semantic Search
    retrieved_chunks = semantic_search(query_embedding, request.top_k)

    # 5. Generate Answer
    answer = generate_answer(request.query, retrieved_chunks)

    return QueryResponse(answer=answer, retrieved_chunks=retrieved_chunks)