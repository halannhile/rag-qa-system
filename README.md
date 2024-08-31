# RAG-based Q&A System

A Retrieval-Augmented Generation (RAG) based Question Answering system that allows users to upload documents and ask questions about their content.

## Features

- Document upload (currently supports PDF files)
- Question answering based on uploaded documents
- Uses LangChain for document processing and question answering
- Integrates with Pinecone for vector storage and retrieval
- Uses OpenAI's language models for embeddings and question answering

## Setup

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with the following content:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENV=your_pinecone_environment
   OPENAI_API_KEY=your_openai_api_key
   ```
   Replace `your_pinecone_api_key`, `your_pinecone_environment`, and `your_openai_api_key` with your actual API keys and environment.

4. Run the application:
   ```
   python app/main.py
   ```

5. Open a web browser and navigate to `http://localhost:5000`

## Usage

1. Use the upload form to upload PDF documents
2. Use the question form to ask questions about the uploaded documents
3. The system will provide answers based on the content of the uploaded documents

## Notes & Reference 

Work in progress...
