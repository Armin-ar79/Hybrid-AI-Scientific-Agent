# 🧠 Hybrid AI Research Agent (RAG + LCEL)

This project develops an Autonomous Research Agent capable of answering complex, specialized questions based on custom uploaded research papers (PDFs).

The system architecture is a crucial example of **Hybrid AI**, combining the statistical pattern recognition of a Large Language Model (LLM) with a structured external knowledge base (Vector Database) for **Symbolic Reasoning**. This ensures factual grounding in the user's specialized domain (Robotic Materials).

## 🌟 Features

- **Retrieval-Augmented Generation (RAG):** Answers are grounded in user-provided PDF research papers.
- **Hybrid Architecture:** Combines an open-source LLM (FLAN-T5) with a Vector Database (ChromaDB).
- **Semantic Search:** Uses Sentence Transformers to retrieve contextually relevant information from documents.
- **Core Technology:** Built using the modern **LangChain Expression Language (LCEL)** for a robust and modular pipeline.
- **Application:** Functions as a private research assistant for complex technical queries.

## 🛠️ Tech Stack

- **Language:** Python
- **LLM/Orchestration:** LangChain, HuggingFacePipeline
- **Vector DB:** ChromaDB
- **Models:** FLAN-T5-small, all-MiniLM-L6-v2
- **Tools:** PyPDF, Transformers

## 🚀 How to Run

### 1. Install Dependencies
```bash
# Install core libraries
pip install langchain langchain-community langchain-huggingface pypdf chromadb sentence-transformers torch

### 2. Prepare Knowledge Base
Place your research papers (.pdf files) inside a folder named docs in the root directory.

### 3. Ingest Data (Build the Brain)
py ingest_docs.py
(This script will create the vector_db folder.)

### 4. Run the Agent
py agent.py
