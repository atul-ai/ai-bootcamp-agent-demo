import json
import os
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import VECTOR_DB_DIR, EMBEDDINGS_MODEL


def ingest_arxiv_data(json_file_path: str) -> None:
    """
    Ingest arxiv data from a JSON file into the vector database.

    Args:
        json_file_path: Path to the JSON file containing arxiv data
    """
    # Load the JSON data
    with open(json_file_path, "r") as file:
        papers = json.load(file)

    # Initialize the embeddings model
    embedding_model = SentenceTransformer(EMBEDDINGS_MODEL)

    # Initialize the Chroma client
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    # Create or get the collection
    try:
        collection = client.get_collection("arxiv_papers")
        print("Using existing collection.")
    except ValueError:
        collection = client.create_collection(
            name="arxiv_papers", metadata={"description": "arXiv papers collection"}
        )
        print("Created new collection.")

    # Process papers in batches
    batch_size = 100
    total_papers = len(papers)

    for i in range(0, total_papers, batch_size):
        batch = papers[i : i + batch_size]

        ids = []
        documents = []
        metadatas = []

        print(
            f"Processing batch {i // batch_size + 1}/{(total_papers - 1) // batch_size + 1}"
        )

        for paper in tqdm(batch):
            paper_id = paper.get("id", "")
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            authors = paper.get("authors", [])

            # Skip papers with missing data
            if not paper_id or not title or not abstract:
                continue

            # Create document text (used for embedding)
            document = f"{title}\n\n{abstract}"

            # Create metadata
            metadata = {
                "title": title,
                "authors": ", ".join(authors) if isinstance(authors, list) else authors,
                "year": paper.get("year"),
                "categories": paper.get("categories", []),
            }

            ids.append(paper_id)
            documents.append(document)
            metadatas.append(metadata)

        # Add batch to collection
        if ids:
            collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"Ingestion complete. Total papers in database: {collection.count()}")


def search_papers(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for papers in the vector database.

    Args:
        query: The search query
        k: Number of results to return

    Returns:
        List of paper metadata
    """
    # Initialize the embeddings model
    embedding_model = SentenceTransformer(EMBEDDINGS_MODEL)

    # Initialize the Chroma client
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    try:
        collection = client.get_collection("arxiv_papers")
    except ValueError:
        return {
            "error": "Vector database is not initialized. Please ingest data first."
        }

    # Generate embedding for the query
    query_embedding = embedding_model.encode(query).tolist()

    # Search for similar papers
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["metadatas", "documents"],
    )

    papers = []
    for i in range(len(results["ids"][0])):
        papers.append(
            {
                "id": results["ids"][0][i],
                "title": results["metadatas"][0][i].get("title", "Unknown"),
                "authors": results["metadatas"][0][i].get("authors", "Unknown"),
                "abstract": results["documents"][0][i],
                "similarity_score": results["distances"][0][i]
                if "distances" in results
                else None,
            }
        )

    return papers
