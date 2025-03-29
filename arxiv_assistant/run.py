import uvicorn
from dotenv import load_dotenv
import os

from arxiv_assistant.utils.config import API_HOST, API_PORT


def main():
    """Run the arXiv Assistant API."""
    port = 8002  # Hardcoded port to ensure it works
    host = "0.0.0.0"
    print(f"Starting arXiv Assistant API on {host}:{port}")
    uvicorn.run(
        "arxiv_assistant.api.app:app", host=host, port=port, reload=True
    )


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Run the application
    main()
