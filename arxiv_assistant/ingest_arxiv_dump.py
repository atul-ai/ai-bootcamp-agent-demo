#!/usr/bin/env python3
"""
Command-line script to ingest an ArXiv dump directly.
Usage: python ingest_arxiv_dump.py path/to/arxiv_dump.json
"""

import sys
import os
from dotenv import load_dotenv
import argparse

from arxiv_assistant.utils.arxiv_ingestion import process_arxiv_dump


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Ingest an ArXiv dump file into the vector database for search and analysis.",
        epilog="""
Examples:
  python -m arxiv_assistant.ingest_arxiv_dump /path/to/arxiv_dump.json
  python -m arxiv_assistant.ingest_arxiv_dump /path/to/arxiv_dump.json --batch_size 200

Expected ArXiv JSON format:
  [
    {
      "id": "paper_id",
      "submitter": "submitter_name",
      "authors": ["Author 1", "Author 2"],
      "title": "Paper Title",
      "comments": "10 pages, 5 figures",
      "journal-ref": "Journal reference",
      "doi": "DOI",
      "abstract": "Paper abstract text",
      "categories": ["cs.AI", "cs.LG"],
      "versions": [...]
    },
    ...
  ]

Notes:
  - For large files, consider creating a sample first with create_sample.py
  - Processing time depends on the file size and batch size
  - Papers are processed in batches to optimize memory usage
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file_path", help="Path to the ArXiv dump JSON file to ingest")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100). Larger batches may be faster but use more memory",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Check if file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File '{args.file_path}' not found.")
        sys.exit(1)

    try:
        print(f"Starting ingestion of ArXiv dump from {args.file_path}")
        process_arxiv_dump(args.file_path, batch_size=args.batch_size)
        print("Ingestion completed successfully.")
    except Exception as e:
        print(f"Error during ingestion: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
