# arXiv Assistant

An AI-powered assistant for searching, downloading, and summarizing academic papers from arXiv.

## Features

- **Search**: Find relevant papers on arXiv using semantic search
- **Download**: Retrieve full PDFs of papers by their arXiv ID
- **Summarize**: Generate concise summaries of papers using LLMs
- **Combined Workflow**: Search for papers by title and get summaries in one step

## Installation

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Setup

1. Clone the repository
   ```bash
   git clone https://github.com/atul-ai/ai-bootcamp-agent-demo.git
   cd AI\ Agent\ Demo
   ```

2. Create and activate a virtual environment
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Usage

### Starting the Server

```bash
# Start the API server
python -m arxiv_assistant.api.app
```

The server will be available at `http://localhost:8002`.

### API Endpoints

- `POST /search_papers`: Search for papers by keywords or query
- `POST /search_by_title`: Find a specific paper by its title
- `GET /paper/{paper_id}`: Get metadata for a specific paper
- `POST /summarize_paper`: Generate a summary for a paper by ID
- `POST /download_paper`: Download a paper PDF by ID
- `POST /search_and_summarize`: Combined endpoint to search and summarize in one step

## Project Structure

```
arxiv_assistant/
├── agents/               # Agent implementations
├── api/                  # FastAPI server and endpoints
├── data/                 # Data storage
│   ├── papers/           # Downloaded paper PDFs
│   ├── temp/             # Temporary files
│   └── vectordb/         # Vector database for search
├── logs/                 # Application logs
├── tools/                # Tools used by agents
└── utils/                # Utility functions
```

## Models

The application uses:
- Groq API for generating summaries
- SambaNova models as an alternative
- Sentence transformers for embeddings and search

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License
The app is built with videcoding, libraries used could be copyrighted. Review the licenses for your usecase before you use them.
All other files are covered by the MIT license, see [LICENSE](https://github.com/atul-ai/ai-bootcamp-agent-demo/blob/main/LICENSE). 
