# LLM_project_new
Project Overview
This project is a Retrieval-Augmented Generation (RAG) based chatbot that allows users to upload PDF documents and ask questions in natural language. The system processes the document and generates accurate answers based only on the uploaded content.
Unlike cloud-based solutions, this system uses Ollama, enabling fully offline execution, improved data privacy, and reduced dependency on external APIs.

How It Works:
Document Ingestion
Load documents from /data
Convert into chunks
Generate embeddings using Ollama
Storage
Store embeddings in Qdrant vector database
Query
User asks a question
Query is embedded
Similar chunks retrieved from Qdrant
Answer Generation
Context + query sent to Ollama LLM
Response generated
Output
Answer displayed with source chunks
