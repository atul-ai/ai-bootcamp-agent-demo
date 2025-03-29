from typing import List, Dict, Any
from langchain.tools import BaseTool
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.errors import InvalidCollectionException
import os

from ..utils.config import VECTOR_DB_DIR, EMBEDDINGS_MODEL


class EmbeddingLookupTool(BaseTool):
    name: str = "embedding_lookup"
    description: str = "Search for relevant papers based on a query"

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self):
        super().__init__()
        self.embedding_model = SentenceTransformer(EMBEDDINGS_MODEL)
        self.client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

        # Check if collection exists
        try:
            self.collection = self.client.get_collection("arxiv_papers")
        except InvalidCollectionException:
            # Collection doesn't exist yet - that's OK for now
            self.collection = None

    def _run(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for papers relevant to the query.

        Args:
            query: The search query
            k: Number of results to return

        Returns:
            List of paper metadata
        """
        if self.collection is None:
            return {
                "error": "Vector database is not initialized. Please ingest data first."
            }

        # Generate embedding for the query
        query_embedding = self.embedding_model.encode(query).tolist()

        # Search for similar papers
        results = self.collection.query(
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
