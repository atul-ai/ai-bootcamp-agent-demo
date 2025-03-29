# arXiv Assistant Frontend

A simple web-based user interface for the arXiv Assistant API.

## Features

- Search for research papers using semantic search
- Get summaries of specific papers
- Chat with an AI assistant about research papers
- Switch between different AI models (Groq and Sambanova)

## Prerequisites

- Python 3.6 or higher
- arXiv Assistant API running on port 8002

## Setup and Usage

1. Make sure the arXiv Assistant API is running on http://localhost:8002
   ```bash
   cd /path/to/project
   source venv/bin/activate
   python -m arxiv_assistant.run
   ```

2. Start the frontend server (in a separate terminal)
   ```bash
   cd /path/to/project/frontend
   ./serve.py
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:3000
   ```

## API Endpoints Used

The frontend interacts with the following API endpoints:

- `GET /` - Check if the API is running
- `POST /search_papers` - Search for papers with a specified query
- `POST /query` - Submit queries to the AI assistant

## Troubleshooting

- If the status indicator shows "API Disconnected", ensure the API is running on port 8002
- If you get CORS errors, ensure the API allows cross-origin requests
- For any other issues, check the browser console for error messages

## License

This project is part of the arXiv Assistant demo. 