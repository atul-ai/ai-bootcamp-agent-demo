#!/usr/bin/env python3
"""
Command-line script to create a sample of a large ArXiv dump file.
Usage: python create_sample.py path/to/arxiv_dump.json --sample_size 1000 --output sample.json
"""

import sys
import os
from dotenv import load_dotenv
import argparse

from arxiv_assistant.utils.arxiv_ingestion import create_arxiv_sample


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Create a random sample from a large ArXiv dump file.",
        epilog="""
Examples:
  python -m arxiv_assistant.create_sample /path/to/arxiv_dump.json
  python -m arxiv_assistant.create_sample /path/to/arxiv_dump.json --sample_size 500
  python -m arxiv_assistant.create_sample /path/to/arxiv_dump.json --output my_sample.json

Notes:
  - This tool uses memory-efficient streaming to handle very large files
  - For a 5GB file, expect processing to take several minutes
  - The tool will show periodic progress updates
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "file_path",
        help="Path to the large ArXiv dump JSON file containing papers data",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Number of papers to include in the sample (default: 1000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the sample file (default: auto-generated in data directory)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Check if file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File '{args.file_path}' not found.")
        sys.exit(1)

    try:
        print(f"Starting to create sample from {args.file_path}")
        output_path = create_arxiv_sample(
            args.file_path, sample_size=args.sample_size, output_file=args.output
        )
        print(f"Sample created successfully at {output_path}")
    except Exception as e:
        print(f"Error creating sample: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
