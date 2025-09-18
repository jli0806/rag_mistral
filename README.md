# RAG Mistral

A Retrieval-Augmented Generation (RAG) system using Mistral AI for building searchable knowledge bases from PDF documents.

## Quick Start

See the [backend documentation](backend/README.md) for detailed setup instructions.

## Project Structure

- `backend/` - FastAPI backend service

## Features

- PDF document ingestion and processing
- Semantic search using Mistral AI embeddings
- Question answering with context-aware responses
- RESTful API with interactive documentation

## Getting Started

1. Clone this repository
2. Follow the setup instructions in the [backend README](backend/README.md)
3. Start the backend service
4. Visit http://localhost:8000/docs for interactive API documentation

## API Endpoints

- **POST /ingest/** - Upload PDF files for ingestion
- **POST /query/** - Submit questions to query the knowledge base

## Technology Stack

- **Backend**: FastAPI, Python
- **AI/ML**: Mistral AI (embeddings and language model)
- **Document Processing**: pypdf for PDF text extraction
- **Vector Search**: Cosine similarity with NumPy/scikit-learn
