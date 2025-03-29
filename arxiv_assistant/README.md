# arXiv Assistant

A simple agent-based system for searching and summarizing academic papers from arXiv.

## Features

- Search for relevant papers based on a query
- Summarize papers using LLM (Deepseek R1 models via Groq/Sambanova)
- Ingest and search standard ArXiv paper dumps
- Category-based filtering of search results
- API for integration with other systems

## Setup

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file from the example:
```bash
cp .env.example .env
```

5. Update the `.env` file with your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
SAMBANOVA_API_KEY=your_sambanova_api_key_here
```

## Data Ingestion

Before using the system, you need to ingest arXiv data into the vector database. There are two options:

### Option 1: Custom Format

Use this if you have a custom JSON file with arXiv papers:

```json
[
  {
    "id": "paper_id",
    "title": "Paper Title",
    "abstract": "Paper abstract text",
    "authors": ["Author 1", "Author 2"],
    "year": 2023,
    "categories": ["cs.AI", "cs.LG"]
  },
  ...
]
```

Call the ingestion endpoint:
```bash
curl -X POST "http://localhost:8000/ingest" -H "Content-Type: application/json" -d '{"file_path": "/path/to/arxiv_data.json"}'
```

### Option 2: Standard ArXiv Dump Format

Use this if you have a standard ArXiv dump with the following fields:
- id: ArXiv ID
- submitter: Who submitted the paper
- authors: Authors of the paper
- title: Title of the paper
- comments: Additional info, such as number of pages and figures
- journal-ref: Information about the journal the paper was published in
- doi: Digital Object Identifier
- abstract: The abstract of the paper
- categories: Categories / tags in the ArXiv system
- versions: A version history

You can ingest the dump in two ways:

1. Using the API:
```bash
curl -X POST "http://localhost:8000/ingest_arxiv_dump" -H "Content-Type: application/json" -d '{"file_path": "/path/to/arxiv_dump.json"}'
```

2. Using the command-line tool:
```bash
# From the project root
python -m arxiv_assistant.ingest_arxiv_dump /path/to/arxiv_dump.json

# You can also adjust the batch size for processing
python -m arxiv_assistant.ingest_arxiv_dump /path/to/arxiv_dump.json --batch_size 200
```

### Handling Large ArXiv Dumps

If you have a very large ArXiv dump file (e.g., 5GB+), you can create a smaller sample for testing:

```bash
# Create a random sample of 1000 papers
python -m arxiv_assistant.create_sample /path/to/large_arxiv_dump.json

# Specify a custom sample size and output file
python -m arxiv_assistant.create_sample /path/to/large_arxiv_dump.json --sample_size 500 --output my_sample.json
```

This uses a memory-efficient streaming approach that can handle very large files without loading the entire dataset into memory.

## Usage

### Starting the API

Start the API:
```bash
python run.py
```

### API Endpoints

#### 1. Process Query (Agent-based)
```
POST /query
```

Request body:
```json
{
  "query": "artificial intelligence",
  "task": "search"  // "search", "summarize", or "auto"
}
```

Response:
```json
{
  "result": "...",
  "status": "success"
}
```

#### 2. Direct Paper Search
```
POST /search_papers
```

Request body:
```json
{
  "query": "transformer architecture",
  "limit": 5,
  "categories": ["cs.AI", "cs.LG"]  // Optional categories filter
}
```

Response:
```json
{
  "papers": [
    {
      "id": "1706.03762",
      "title": "Attention Is All You Need",
      "authors": "Ashish Vaswani, Noam Shazeer, Niki Parmar, ...",
      "abstract": "...",
      "categories": "cs.CL, cs.LG",
      "journal_ref": "",
      "doi": "",
      "comments": "17 pages, 5 figures",
      "similarity_score": 0.89
    },
    ...
  ],
  "status": "success"
}
```

#### 3. Ingest Custom Data
```
POST /ingest
```

Request body:
```json
{
  "file_path": "/path/to/arxiv_data.json"
}
```

#### 4. Ingest ArXiv Dump
```
POST /ingest_arxiv_dump
```

Request body:
```json
{
  "file_path": "/path/to/arxiv_dump.json"
}
```

## Architecture

- **API Layer**: FastAPI application with endpoints for queries and data ingestion
- **Orchestrator Agent**: Routes queries to specialized agents
- **Search Agent**: Finds relevant papers using vector search
- **Summarization Agent**: Downloads and summarizes papers
- **Tools**: Vector search, paper download, and summarization tools
- **Database**: In-memory vector database for paper embeddings

## Development

1. To add new features, extend the existing agent architecture
2. To support new sources, add new tools and update the data ingestion logic
3. To use different LLMs, update the configuration in `utils/config.py`

## Data Sources

You can obtain ArXiv datasets from various sources:
- [Kaggle ArXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- [ArXiv Dataset on Hugging Face](https://huggingface.co/datasets/arxiv_dataset)
- Direct API access: https://arxiv.org/help/api 