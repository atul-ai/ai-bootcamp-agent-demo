#!/usr/bin/env python3
"""
Script to directly ingest a sample arxiv JSON file into the vector database.
"""

import sys
import os
import json
import time
from dotenv import load_dotenv
import argparse
from sentence_transformers import SentenceTransformer
import torch
import chromadb
from chromadb.errors import InvalidCollectionException
from tqdm import tqdm

from arxiv_assistant.utils.config import VECTOR_DB_DIR, EMBEDDINGS_MODEL, DATA_DIR


def ingest_sample(sample_file_path: str, batch_size: int = 50):
    """
    Ingest a sample of papers from a JSON file into the vector database.

    Args:
        sample_file_path: Path to the sample JSON file
        batch_size: Batch size for processing
    """
    print(f"Loading sample file from {sample_file_path}")

    # Load the JSON data
    with open(sample_file_path, "r", encoding="utf-8") as f:
        try:
            papers = json.load(f)
            print(f"Loaded {len(papers)} papers from sample file")
        except json.JSONDecodeError:
            # Try as JSONL
            f.seek(0)
            papers = []
            for line in f:
                if not line.strip():
                    continue
                try:
                    papers.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
            print(f"Loaded {len(papers)} papers from JSONL format")

    # Initialize the embeddings model
    print(f"Initializing embedding model: {EMBEDDINGS_MODEL}")
    # Make sure PyTorch uses CPU
    device = torch.device("cpu")
    embedding_model = SentenceTransformer(EMBEDDINGS_MODEL, device=device)

    # Initialize the Chroma client
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    # Create or get the collection
    try:
        collection = client.get_collection("arxiv_papers")
        print(f"Using existing collection with {collection.count()} papers")
    except (ValueError, InvalidCollectionException):
        collection = client.create_collection(
            name="arxiv_papers", metadata={"description": "arXiv papers collection"}
        )
        print("Created new collection")

    # Process papers in batches
    total_papers = len(papers)

    print(f"Starting ingestion of {total_papers} papers in batches of {batch_size}")
    start_time = time.time()

    for i in range(0, total_papers, batch_size):
        batch = papers[i : i + batch_size]

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        print(
            f"\nProcessing batch {i // batch_size + 1}/{(total_papers - 1) // batch_size + 1}"
        )

        for paper in tqdm(batch):
            # Extract fields with fallbacks
            paper_id = paper.get("id", "")
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")

            # Skip papers with missing essential data
            if not paper_id or not title or not abstract:
                continue

            # Process fields
            authors = paper.get("authors", "")
            if isinstance(authors, list):
                authors = ", ".join(authors)

            categories = paper.get("categories", "")
            if isinstance(categories, list):
                categories = ", ".join(categories)

            # Create document text (used for embedding)
            document = f"Title: {title}\n\nAbstract: {abstract}"

            # Create metadata
            metadata = {
                "title": title or "",
                "authors": authors or "",
                "submitter": paper.get("submitter", "") or "",
                "comments": paper.get("comments", "") or "",
                "journal_ref": paper.get("journal-ref", "") or "",
                "doi": paper.get("doi", "") or "",
                "categories": categories or "",
            }

            ids.append(paper_id)
            documents.append(document)
            metadatas.append(metadata)
            
            # Generate embedding
            with torch.no_grad():
                embedding = embedding_model.encode(document, show_progress_bar=False)
                embeddings.append(embedding.tolist())

        # Add batch to collection
        if ids:
            print(
                f"Adding {len(ids)} papers to vector database"
            )
            collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    elapsed = time.time() - start_time
    print(f"\nIngestion complete in {elapsed:.2f} seconds.")
    print(f"Total papers in database: {collection.count()}")

    # Do a quick test search
    print("\nTesting search functionality with a sample query...")
    with torch.no_grad():
        query_embedding = embedding_model.encode("machine learning").tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        include=["metadatas", "documents"],
    )

    print("\nTop 3 search results for 'machine learning':")
    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        print(f"Paper ID: {results['ids'][0][i]}")
        print(f"Title: {metadata.get('title', 'Unknown')}")
        print(f"Authors: {metadata.get('authors', 'Unknown')}")
        print(f"Categories: {metadata.get('categories', '')}")
        print("-" * 50)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Ingest a sample arxiv JSON/JSONL file directly into the vector database.",
        epilog="""
Examples:
  python -m arxiv_assistant.ingest_sample /path/to/sample.json
  python -m arxiv_assistant.ingest_sample /path/to/sample.json --batch_size 100
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file_path", help="Path to the sample JSON file to ingest")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Batch size for processing (default: 50). Smaller batch sizes use less memory.",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Check if file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File '{args.file_path}' not found.")
        sys.exit(1)

    try:
        ingest_sample(args.file_path, batch_size=args.batch_size)
        print("\nSample ingestion completed successfully.")
    except Exception as e:
        print(f"Error during ingestion: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
