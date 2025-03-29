import os
# Set tokenizers parallelism to avoid deadlocks with forked processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import random
from typing import List, Dict, Any
import chromadb
from chromadb.errors import InvalidCollectionException
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import ijson  # For streaming JSON parsing
import time

from .config import VECTOR_DB_DIR, EMBEDDINGS_MODEL, DATA_DIR
from .logger import get_logger, get_search_logger, get_summary_logger, log_exception

# Get loggers
logger = get_logger(__name__)
search_logger = get_search_logger()
summary_logger = get_summary_logger()


def create_arxiv_sample(
    json_file_path: str, sample_size: int = 1000, output_file: str = None
) -> str:
    """
    Create a random sample of papers from a large ArXiv JSON dump file.
    Handles both standard JSON array format and JSONL (JSON Lines) format.
    Uses streaming to handle very large files efficiently.

    Args:
        json_file_path: Path to the large JSON file
        sample_size: Number of papers to include in the sample
        output_file: Path to save the sample (defaults to data_dir/arxiv_sample_{sample_size}.json)

    Returns:
        Path to the created sample file
    """
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"File not found: {json_file_path}")

    # Default output file if not provided
    if output_file is None:
        filename = os.path.basename(json_file_path)
        name, ext = os.path.splitext(filename)
        output_file = os.path.join(DATA_DIR, f"{name}_sample_{sample_size}{ext}")

    # Use improved reservoir sampling without counting first
    # (This is more efficient for very large files)
    print(
        f"Processing file {json_file_path} to create a sample of {sample_size} papers..."
    )
    print("This may take a while for large files. Please be patient.")

    sample = []
    processed_count = 0
    start_time = time.time()

    try:
        # First determine if it's JSONL format or standard JSON array
        with open(json_file_path, "r", encoding="utf-8") as f:
            first_char = f.read(1).strip()
            is_jsonl = first_char != "["

        if is_jsonl:
            # Handle JSONL format (each line is a JSON object)
            print("Detected JSONL format (each line is a separate JSON object)")
            with open(json_file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue  # Skip empty lines

                    try:
                        item = json.loads(line.strip())
                        if i < sample_size:
                            sample.append(item)
                        else:
                            # For later items, randomly decide whether to include
                            j = random.randint(0, i)
                            if j < sample_size:
                                sample[j] = item

                        processed_count = i + 1
                        if (
                            processed_count % 10000 == 0
                            or (time.time() - start_time) > 15
                        ):
                            elapsed = time.time() - start_time
                            papers_per_sec = (
                                processed_count / elapsed if elapsed > 0 else 0
                            )
                            print(
                                f"Processed {processed_count} papers ({papers_per_sec:.1f} papers/sec)"
                            )
                            start_time = time.time()
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON on line {i + 1}: {e}")
                        continue  # Skip this line and continue
        else:
            # Handle standard JSON array format
            print("Detected standard JSON array format")
            with open(json_file_path, "rb") as f:
                # Use ijson for streaming
                for i, item in enumerate(ijson.items(f, "item")):
                    if i < sample_size:
                        sample.append(item)
                    else:
                        # For later items, randomly decide whether to include
                        j = random.randint(0, i)
                        if j < sample_size:
                            sample[j] = item

                    processed_count = i + 1
                    if processed_count % 10000 == 0 or (time.time() - start_time) > 15:
                        elapsed = time.time() - start_time
                        papers_per_sec = processed_count / elapsed if elapsed > 0 else 0
                        print(
                            f"Processed {processed_count} papers ({papers_per_sec:.1f} papers/sec)"
                        )
                        start_time = time.time()

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        if processed_count > 0:
            print(f"Processed {processed_count} papers before error.")
            print("Continuing with the papers collected so far.")
        else:
            raise

    # Write sample to output file as a JSON array
    print(f"Writing sample of {len(sample)} papers to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)

    print(f"Sample created successfully at {output_file}")
    return output_file


def process_arxiv_dump(json_file_path: str, batch_size: int = 100) -> None:
    """
    Process and ingest an ArXiv JSON dump into the vector database.

    Args:
        json_file_path: Path to the JSON file containing ArXiv data with fields:
            id, submitter, authors, title, comments, journal-ref, doi, abstract, categories, versions
    """
    print(f"Reading ArXiv dump from {json_file_path}...")

    # Load the JSON data
    with open(json_file_path, "r") as file:
        papers = json.load(file)

    print(f"Loaded {len(papers)} papers from file")

    # Initialize the embeddings model
    print(f"Initializing embedding model: {EMBEDDINGS_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDINGS_MODEL)

    # Initialize the Chroma client
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    # Create or get the collection
    try:
        collection = client.get_collection("arxiv_papers")
        print(f"Using existing collection with {collection.count()} papers")
    except ValueError:
        collection = client.create_collection(
            name="arxiv_papers", metadata={"description": "arXiv papers collection"}
        )
        print("Created new collection")

    # Process papers in batches
    batch_size = batch_size
    total_papers = len(papers)

    print(f"Starting ingestion of {total_papers} papers in batches of {batch_size}")

    for i in range(0, total_papers, batch_size):
        batch = papers[i : i + batch_size]

        ids = []
        documents = []
        metadatas = []

        print(
            f"Processing batch {i // batch_size + 1}/{(total_papers - 1) // batch_size + 1}"
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

            # Extract version info
            versions = paper.get("versions", [])
            latest_version = versions[-1] if versions else {}
            version_date = latest_version.get("created", "") if versions else ""

            # Create document text (used for embedding)
            document = f"Title: {title}\n\nAbstract: {abstract}"

            # Create metadata
            metadata = {
                "title": title,
                "authors": authors,
                "submitter": paper.get("submitter", ""),
                "comments": paper.get("comments", ""),
                "journal_ref": paper.get("journal-ref", ""),
                "doi": paper.get("doi", ""),
                "categories": categories,
                "version_date": version_date,
                "num_versions": len(versions) if isinstance(versions, list) else 0,
            }

            ids.append(paper_id)
            documents.append(document)
            metadatas.append(metadata)

        # Add batch to collection
        if ids:
            print(f"Adding {len(ids)} papers to vector database")
            collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"Ingestion complete. Total papers in database: {collection.count()}")


def search_arxiv_papers(
    query: str, k: int = 5, filter_categories: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for papers in the vector database with optional category filtering.

    Args:
        query: The search query
        k: Number of results to return
        filter_categories: Optional list of categories to filter by

    Returns:
        List of paper metadata
    """
    try:
        search_logger.info(f"Search query: '{query}', k={k}, categories={filter_categories}")
        
        # Import torch here to avoid loading it unnecessarily
        import torch
        
        # Initialize the embeddings model
        device = torch.device("cpu")
        embedding_model = SentenceTransformer(EMBEDDINGS_MODEL, device=device)

        # Initialize the Chroma client
        client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

        try:
            collection = client.get_collection("arxiv_papers")
        except (ValueError, InvalidCollectionException) as e:
            error_msg = "Vector database is not initialized. Please ingest data first."
            search_logger.error(error_msg)
            return {
                "error": error_msg
            }

        # Generate embedding for the query
        with torch.no_grad():
            query_embedding = embedding_model.encode(query).tolist()
            search_logger.info(f"Generated embedding for query (dimensions: {len(query_embedding)})")

        # Prepare filter if categories are specified
        where_clause = None
        if filter_categories:
            where_clause = {
                "$or": [{"categories": {"$contains": cat}} for cat in filter_categories]
            }
            search_logger.info(f"Applied category filter: {where_clause}")

        # Search for similar papers
        results = collection.query(
            query_embeddings=[query_embedding],  # Ensure this is a list
            n_results=k,
            where=where_clause,
            include=["metadatas", "documents"],
        )

        papers = []
        
        # Check if results are empty
        if not results or not results.get("ids") or len(results["ids"]) == 0:
            search_logger.warning(f"Empty results or missing ids: {results}")
            return papers
            
        search_logger.info(f"Found {len(results['ids'][0])} matching papers")
            
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]
            paper_content = results["documents"][0][i] if results.get("documents") and len(results["documents"]) > 0 else ""
            paper_id = results["ids"][0][i] if results.get("ids") and len(results["ids"]) > 0 else "Unknown"
            
            # Safely extract abstract from the document
            abstract = ""
            if paper_content:
                parts = paper_content.split("\n\nAbstract: ")
                if len(parts) > 1:
                    abstract = parts[1]
                else:
                    abstract = paper_content
            
            # Safely calculate similarity score
            similarity_score = None
            if results and "distances" in results and results["distances"] and len(results["distances"]) > 0 and len(results["distances"][0]) > i:
                similarity_score = results["distances"][0][i]
            
            papers.append(
                {
                    "id": paper_id,
                    "title": metadata.get("title", "Unknown") if metadata else "Unknown",
                    "authors": metadata.get("authors", "Unknown") if metadata else "Unknown",
                    "abstract": abstract,
                    "categories": metadata.get("categories", "") if metadata else "",
                    "journal_ref": metadata.get("journal_ref", "") if metadata else "",
                    "doi": metadata.get("doi", "") if metadata else "",
                    "comments": metadata.get("comments", "") if metadata else "",
                    "similarity_score": similarity_score,
                }
            )
        
        search_logger.info(f"Returning {len(papers)} papers for query: '{query}'")
        return papers
        
    except Exception as e:
        error_msg = log_exception(logger, "Error in search_arxiv_papers", e)
        search_logger.error(f"Search failed: {error_msg}")
        return {
            "error": f"An error occurred during search: {str(e)}"
        }


def get_paper_by_id(paper_id: str) -> Dict[str, Any]:
    """
    Retrieve a paper from the vector database by its ID.

    Args:
        paper_id: The ID of the paper to retrieve

    Returns:
        Paper metadata and content or error message
    """
    try:
        logger.info(f"Retrieving paper with ID: {paper_id}")
        
        # Initialize the Chroma client
        client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

        try:
            collection = client.get_collection("arxiv_papers")
        except (ValueError, InvalidCollectionException) as e:
            error_msg = "Vector database is not initialized. Please ingest data first."
            logger.error(error_msg)
            return {"error": error_msg}

        # Query for the specific paper by ID
        results = collection.get(
            ids=[paper_id],
            include=["metadatas", "documents"]
        )
        
        # Check if paper was found
        if not results or not results.get("ids") or len(results["ids"]) == 0:
            error_msg = f"Paper with ID {paper_id} not found"
            logger.warning(error_msg)
            return {"error": error_msg}
            
        logger.info(f"Found paper with ID: {paper_id}")
        
        # Extract paper data
        metadata = results["metadatas"][0] if results.get("metadatas") and len(results["metadatas"]) > 0 else {}
        paper_content = results["documents"][0] if results.get("documents") and len(results["documents"]) > 0 else ""
        
        # Safely extract abstract from the document
        abstract = ""
        if paper_content:
            parts = paper_content.split("\n\nAbstract: ")
            if len(parts) > 1:
                abstract = parts[1]
            else:
                abstract = paper_content
        
        # Construct paper object
        paper = {
            "id": paper_id,
            "title": metadata.get("title", "Unknown") if metadata else "Unknown",
            "authors": metadata.get("authors", "Unknown") if metadata else "Unknown",
            "abstract": abstract,
            "categories": metadata.get("categories", "") if metadata else "",
            "journal_ref": metadata.get("journal_ref", "") if metadata else "",
            "doi": metadata.get("doi", "") if metadata else "",
            "comments": metadata.get("comments", "") if metadata else "",
        }
        
        return paper
        
    except Exception as e:
        error_msg = log_exception(logger, f"Error retrieving paper with ID {paper_id}", e)
        return {"error": error_msg}

async def summarize_paper(paper_id: str, model: str = "groq") -> Dict[str, Any]:
    """
    Generate a summary for a specific paper.

    Args:
        paper_id: The ID of the paper to summarize
        model: The model to use for summarization ("groq" or "sambanova")

    Returns:
        Paper summary and metadata or error message
    """
    try:
        summary_logger.info(f"Summarization request for paper ID: {paper_id} using model: {model}")
        
        # First retrieve the paper
        paper = get_paper_by_id(paper_id)
        
        # Check if there was an error retrieving the paper
        if "error" in paper:
            summary_logger.error(f"Error retrieving paper: {paper['error']}")
            return paper
        
        summary_logger.info(f"Generating summary for paper: {paper_id}, title: '{paper.get('title', 'Unknown')}'")
        
        # Download the full paper using the ArxivPaperDownloaderTool
        from ..tools.paper_downloader import ArxivPaperDownloaderTool
        downloader = ArxivPaperDownloaderTool()
        download_result = await downloader._arun(paper_id)
        
        # Use the downloaded paper if available
        paper_content = None
        if download_result.get("status") == "success" and "content" in download_result:
            summary_logger.info(f"Using downloaded paper content from: {download_result.get('file_path', 'unknown')}")
            paper_content = download_result["content"]
            summary_logger.info(f"Retrieved {len(paper_content)} characters of paper content")
        else:
            summary_logger.warning(f"Could not retrieve paper content: {download_result.get('message', 'Unknown error')}")
        
        # Create a summarization request to the LLM
        from ..agents.orchestrator_agent import OrchestratorAgent
        
        # Initialize the appropriate agent
        use_sambanova = model.lower() == "sambanova"
        orchestrator = OrchestratorAgent(use_sambanova=use_sambanova)
        
        # Build prompt with paper information
        title = paper.get("title", "")
        authors = paper.get("authors", "")
        abstract = paper.get("abstract", "")
        
        # Use the agent to generate the summary
        if paper_content:
            query = f"""Summarize this paper based on the full text: 
            Title: {title}
            Authors: {authors}
            Abstract: {abstract}
            
            Full Paper Content:
            {paper_content}
            """
            summary_logger.info(f"Sending query with full paper content to {model} model")
        else:
            query = f"Summarize this paper: Title: {title}, Authors: {authors}, Abstract: {abstract}"
            summary_logger.info(f"Sending query with abstract only to {model} model")
        
        # This will be an async function call if we're in an async context
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context
                result = asyncio.create_task(orchestrator.process(query))
                # Need to await the task
                result = await result
            else:
                # We're not in an async context
                result = asyncio.run(orchestrator.process(query))
        except RuntimeError:
            # No event loop, so create one
            result = asyncio.run(orchestrator.process(query))
        
        # Add the original paper data to the result
        if "result" in result:
            summary_logger.info(f"Successfully generated summary for paper ID: {paper_id}")
            summary_logger.debug(f"Generated summary content: {result['result']}")
            return {
                "paper": paper,
                "summary": result["result"],
                "status": "success"
            }
        else:
            error = result.get("error", "Unknown error occurred during summarization")
            summary_logger.error(f"Failed to generate summary: {error}")
            return {
                "paper": paper,
                "error": error,
                "status": "error"
            }
        
    except Exception as e:
        error_msg = log_exception(logger, f"Error summarizing paper with ID {paper_id}", e)
        summary_logger.error(f"Exception during summarization: {error_msg}")
        return {"error": error_msg, "status": "error"}


def search_paper_by_title(title: str) -> Dict[str, Any]:
    """
    Search for a paper by its title and return the best match.
    
    Args:
        title: The title to search for
        
    Returns:
        The best matching paper or error message
    """
    try:
        summary_logger.info(f"Searching for paper with title similar to: '{title}'")
        
        # Use the search function with the title as query
        results = search_arxiv_papers(query=title, k=3)
        
        # If search returned an error
        if isinstance(results, dict) and "error" in results:
            summary_logger.error(f"Search error: {results['error']}")
            return results
            
        # If no results were found
        if not results or len(results) == 0:
            error_msg = f"No papers found matching title: {title}"
            summary_logger.warning(error_msg)
            return {"error": error_msg}
            
        summary_logger.info(f"Found {len(results)} potential matches for title: '{title}'")
        
        # Find the best match based on title similarity
        best_match = None
        best_score = 0
        
        for paper in results:
            paper_title = paper.get("title", "").lower()
            search_title = title.lower()
            
            # Calculate simple word overlap score
            title_words = set(search_title.split())
            paper_words = set(paper_title.split())
            common_words = title_words.intersection(paper_words)
            
            if len(title_words) > 0:
                score = len(common_words) / len(title_words)
                
                # If perfect or very good match, return immediately
                if score > 0.8:
                    summary_logger.info(f"Found high-quality match: {paper.get('id')} - '{paper.get('title')}'")
                    return paper
                
                # Otherwise keep track of best match
                if score > best_score:
                    best_score = score
                    best_match = paper
        
        # Return the best match if it exists
        if best_match:
            summary_logger.info(f"Best match: {best_match.get('id')} - '{best_match.get('title')}' (score: {best_score:.2f})")
            return best_match
        
        # If no good match was found
        error_msg = f"No good matches found for title: {title}"
        summary_logger.warning(error_msg)
        return {"error": error_msg}
        
    except Exception as e:
        error_msg = log_exception(logger, f"Error searching for paper by title: {title}", e)
        summary_logger.error(f"Title search failed: {error_msg}")
        return {"error": error_msg}


async def search_and_summarize(title: str, model: str = "groq") -> Dict[str, Any]:
    """
    Search for a paper by title, retrieve it, and generate a summary.
    
    This function combines multiple steps:
    1. Search for a paper by title
    2. Get the paper's ID
    3. Generate a summary using the specified model
    
    Args:
        title: The title of the paper to search for
        model: The model to use for summarization ("groq" or "sambanova")
        
    Returns:
        Paper data and summary or error message
    """
    try:
        summary_logger.info(f"Starting search-and-summarize flow for title: '{title}'")
        
        # Step 1: Search for the paper by title
        paper = search_paper_by_title(title)
        
        # Check if search returned an error
        if "error" in paper:
            return paper
        
        # Step 2: Get the paper ID
        paper_id = paper.get("id")
        if not paper_id:
            error_msg = "Found paper but it has no ID"
            summary_logger.error(error_msg)
            return {"error": error_msg}
        
        summary_logger.info(f"Found paper ID: {paper_id}, proceeding to summarization")
        
        # Step 3: Generate summary using the found paper ID
        summary_result = await summarize_paper(paper_id=paper_id, model=model)
        
        # Return the combined result
        return summary_result
        
    except Exception as e:
        error_msg = log_exception(logger, f"Error in search-and-summarize flow for title: {title}", e)
        summary_logger.error(f"Search-and-summarize failed: {error_msg}")
        return {"error": error_msg, "status": "error"}


class Config:
    arbitrary_types_allowed = True
    extra = "allow"
