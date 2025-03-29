# Basic Agent Framework PRD

## Overview
This document outlines the requirements for a streamlined agent framework that helps users search for and summarize academic papers from arxiv. The system leverages LangGraph for agent orchestration and LLM interfaces provided by Groq (Deepseek R1 Distil Llama 70B) and Sambanova (Deepseek R1 671B).

## Problem Statement
Researchers need an efficient way to find relevant papers and obtain summaries without manually searching through vast repositories of academic literature.

## Core Functionality

### 1. Data Ingestion
- Ingest arxiv abstract dataset
- Generate embeddings for titles, abstracts, authors, and IDs
- Store embeddings in an in-memory vector database (Chroma or FAISS)

### 2. User Queries
- Accept paper name to find relevant papers
- Accept paper name to summarize the paper
- Return search results or summaries based on the query

### 3. Tool Suite
- **Embedding Lookup Tool**: Search for relevant papers based on semantic similarity
- **Paper Downloader Tool**: Download papers from arxiv based on paper ID
- **Summarization Tool**: Generate summaries of papers using LLM

### 4. Agent Architecture
- **Central Orchestrator**: Decide next steps based on user queries
- **Search Agent**: Find relevant papers using embedding lookup
- **Summarization Agent**: Generate summaries of papers

## Technical Architecture

### 1. API Service
```
+-------------------+
| API Endpoints     |
+-------------------+
         |
+-------------------+
| Agent Orchestrator|
+-------------------+
         |
+--------+----------+
|                   |
v                   v
+-------------+    +-----------------+
| Search      |    | Summarization   |
| Agent       |    | Agent           |
+-------------+    +-----------------+
      |                    |
      v                    v
+-------------+    +-----------------+
| Vector DB   |    | LLM Interface   |
+-------------+    +-----------------+
```

### 2. Components
- **API Service**: Flask/FastAPI based service
- **Vector Database**: In-memory vector store (Chroma/FAISS)
- **LLM Interface**: 
  - Groq API (Deepseek R1 Distil Llama 70B)
  - Sambanova API (Deepseek R1 671B)
- **Agent Framework**: LangGraph for orchestration

### 3. Data Flow
1. User submits query to API
2. Orchestrator agent determines query type
3. For paper search: Search agent queries vector database
4. For paper summary: Downloader gets paper, summarizer generates summary
5. Results returned to user

## Implementation Requirements

### 1. Technology Stack
- Python 3.8+
- LangChain and LangGraph
- FastAPI for API service
- Chroma or FAISS for vector database

### 2. APIs and Interfaces
- **Query Endpoint**: `/query` (POST)
  - Input: `{"query": "paper_name", "task": "search|summarize"}`
  - Output: `{"results": [...], "status": "success|error"}`

### 3. Data Storage
- In-memory vector database for embeddings
- Temporary storage for downloaded papers

### 4. Performance Criteria
- Focus on accuracy of search results and summaries
- Reasonable response times (acceptable given LLM latency)

## Development Phases

### Phase 1: Core Infrastructure
1. Set up API service
2. Implement embedding generation
3. Set up vector database

### Phase 2: Tools & Agents
1. Implement paper search tool
2. Implement paper download tool
3. Implement summarization tool
4. Create basic agent structure

### Phase 3: Integration & Testing
1. Connect agents via LangGraph
2. Implement orchestration logic
3. Test with sample queries
4. Optimize for accuracy

## Next Steps
1. Set up development environment
2. Create initial API endpoints
3. Implement embedding generation
4. Build and test individual tools
5. Develop agent orchestration 