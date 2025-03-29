import os
# Set tokenizers parallelism to avoid deadlocks with forked processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json

from ..agents.orchestrator_agent import OrchestratorAgent
from ..utils.data_ingestion import ingest_arxiv_data
from ..utils.arxiv_ingestion import process_arxiv_dump, search_arxiv_papers, summarize_paper, get_paper_by_id, search_paper_by_title, search_and_summarize
from ..utils.logger import get_logger, log_exception
from ..tools.paper_downloader import ArxivPaperDownloaderTool

# Configure logger
logger = get_logger("api")

app = FastAPI(title="arXiv Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create the orchestrator agent - we'll instantiate it when needed
# orchestrator = OrchestratorAgent()


class QueryRequest(BaseModel):
    """Request model for user queries."""

    query: str
    task: Optional[str] = "auto"  # "search", "summarize", or "auto"
    model: Optional[str] = "groq"  # "groq" or "sambanova"


class DataIngestionRequest(BaseModel):
    """Request model for data ingestion."""

    file_path: str


class ArxivDumpRequest(BaseModel):
    """Request model for ArXiv dump ingestion."""

    file_path: str


class SearchRequest(BaseModel):
    """Request model for direct paper search."""

    query: str
    limit: Optional[int] = 5
    categories: Optional[List[str]] = None
    model: Optional[str] = "groq"  # "groq" or "sambanova"


class SummaryRequest(BaseModel):
    """Request model for paper summarization."""

    paper_id: str
    model: Optional[str] = "groq"  # "groq" or "sambanova"


class SearchByTitleRequest(BaseModel):
    """Request model for paper search by title."""
    title: str


class SearchAndSummarizeRequest(BaseModel):
    """Request model for search and summarize flow."""
    title: str
    model: str = "groq"


class DownloadRequest(BaseModel):
    paper_id: str


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "arXiv Assistant API is running"}


@app.post("/query")
async def process_query(request: QueryRequest):
    """
    Process a user query.

    Args:
        request: The query request

    Returns:
        Results of processing the query
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")

    # Create orchestrator with requested model
    use_sambanova = request.model.lower() == "sambanova"
    orchestrator = OrchestratorAgent(use_sambanova=use_sambanova)

    # Process query with orchestrator
    if request.task == "search":
        query = f"Search for papers about: {request.query}"
    elif request.task == "summarize":
        query = f"Summarize the paper with ID {request.query}"
    else:  # auto
        query = request.query

    result = await orchestrator.process(query)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@app.post("/ingest")
async def ingest_data(request: DataIngestionRequest, background_tasks: BackgroundTasks):
    """
    Ingest data from a JSON file.

    Args:
        request: The data ingestion request

    Returns:
        Status of the ingestion process
    """
    if not os.path.exists(request.file_path):
        raise HTTPException(
            status_code=400, detail=f"File not found: {request.file_path}"
        )

    try:
        # Run ingestion in the background
        background_tasks.add_task(ingest_arxiv_data, request.file_path)
        return {"message": "Ingestion started in the background", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest_arxiv_dump")
async def ingest_arxiv_dump(
    request: ArxivDumpRequest, background_tasks: BackgroundTasks
):
    """
    Ingest an ArXiv dump file with the standard ArXiv format.

    Args:
        request: The ArXiv dump ingestion request

    Returns:
        Status of the ingestion process
    """
    if not os.path.exists(request.file_path):
        raise HTTPException(
            status_code=400, detail=f"File not found: {request.file_path}"
        )

    try:
        # Run ingestion in the background
        background_tasks.add_task(process_arxiv_dump, request.file_path)
        return {
            "message": "ArXiv dump ingestion started in the background",
            "status": "success",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search_papers")
async def search_papers(request: SearchRequest):
    """
    Directly search for papers using vector search.

    Args:
        request: The search request

    Returns:
        List of matching papers
    """
    if not request.query:
        logger.warning("Search request received with empty query")
        raise HTTPException(status_code=400, detail="Query is required")

    try:
        logger.info(f"Search request: query='{request.query}', limit={request.limit}, model={request.model}")
        results = search_arxiv_papers(
            query=request.query, k=request.limit, filter_categories=request.categories
        )

        if isinstance(results, dict) and "error" in results:
            error_msg = results["error"]
            logger.error(f"Search error: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

        logger.info(f"Search successful: {len(results)} papers found")
        return {"papers": results, "status": "success"}
    except Exception as e:
        error_msg = log_exception(logger, "Error in search_papers endpoint", e)
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/summarize_paper")
async def get_paper_summary(request: SummaryRequest):
    """
    Generate a summary for a specific paper.

    Args:
        request: The summary request containing paper_id

    Returns:
        Paper summary and metadata
    """
    if not request.paper_id:
        logger.warning("Summary request received with empty paper_id")
        raise HTTPException(status_code=400, detail="Paper ID is required")

    try:
        logger.info(f"Summary request for paper ID: {request.paper_id}, model: {request.model}")
        result = await summarize_paper(
            paper_id=request.paper_id,
            model=request.model
        )

        if "error" in result:
            logger.error(f"Summary error: {result['error']}")
            raise HTTPException(status_code=500, detail=result["error"])

        logger.info(f"Summary generated successfully for paper ID: {request.paper_id}")
        if "summary" in result:
            logger.info(f"Summary content: {result['summary']}")
        return result
    except Exception as e:
        error_msg = log_exception(logger, f"Error in summarize_paper endpoint for paper ID: {request.paper_id}", e)
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/paper/{paper_id}")
async def get_paper(paper_id: str):
    """
    Retrieve a paper by its ID.

    Args:
        paper_id: The ID of the paper to retrieve

    Returns:
        Paper metadata and content
    """
    if not paper_id:
        logger.warning("Paper request received with empty paper_id")
        raise HTTPException(status_code=400, detail="Paper ID is required")

    try:
        logger.info(f"Paper request for ID: {paper_id}")
        result = get_paper_by_id(paper_id=paper_id)

        if "error" in result:
            logger.error(f"Paper retrieval error: {result['error']}")
            raise HTTPException(status_code=404 if "not found" in result["error"].lower() else 500, detail=result["error"])

        logger.info(f"Paper retrieved successfully: {paper_id}")
        return {"paper": result, "status": "success"}
    except Exception as e:
        error_msg = log_exception(logger, f"Error in get_paper endpoint for paper ID: {paper_id}", e)
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/search_by_title")
async def search_by_title(request: SearchByTitleRequest):
    """
    Search for a paper by its title and return the best match.

    Args:
        request: The search by title request containing the title

    Returns:
        Best matching paper or error message
    """
    if not request.title:
        logger.warning("Title search request received with empty title")
        raise HTTPException(status_code=400, detail="Paper title is required")

    try:
        logger.info(f"Searching for paper with title: '{request.title}'")
        result = search_paper_by_title(request.title)

        if "error" in result:
            logger.error(f"Title search error: {result['error']}")
            raise HTTPException(status_code=404, detail=result["error"])

        logger.info(f"Found paper with title: '{result.get('title')}', ID: {result.get('id')}")
        return {"paper": result, "status": "success"}
    except Exception as e:
        error_msg = log_exception(logger, f"Error in search_by_title endpoint for title: {request.title}", e)
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/search_and_summarize")
async def search_and_summarize_paper(request: SearchAndSummarizeRequest):
    """
    Search for a paper by title, retrieve it, and generate a summary.

    Args:
        request: The request containing the paper title and model to use

    Returns:
        Paper data with summary or error message
    """
    if not request.title:
        logger.warning("Search and summarize request received with empty title")
        raise HTTPException(status_code=400, detail="Paper title is required")

    try:
        logger.info(f"Processing search and summarize request for title: '{request.title}', model: {request.model}")
        result = await search_and_summarize(request.title, request.model)

        if "error" in result:
            logger.error(f"Search and summarize error: {result['error']}")
            raise HTTPException(status_code=404 if "No papers found" in result["error"] else 500, detail=result["error"])

        logger.info(f"Successfully completed search and summarize for title: '{request.title}'")
        if "summary" in result:
            logger.debug(f"Summary content for paper '{request.title}': {result['summary']}")
        return result
    except Exception as e:
        error_msg = log_exception(logger, f"Error in search_and_summarize endpoint for title: {request.title}", e)
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/download_paper")
async def download_paper(request: DownloadRequest):
    """
    Download a paper from arXiv by its ID.

    Args:
        request: The download request containing paper_id

    Returns:
        Paper metadata and download information
    """
    if not request.paper_id:
        logger.warning("Download request received with empty paper_id")
        raise HTTPException(status_code=400, detail="Paper ID is required")

    try:
        logger.info(f"Download request for paper ID: {request.paper_id}")
        
        # Initialize the paper downloader tool
        downloader = ArxivPaperDownloaderTool()
        
        # Download the paper
        result = await downloader._arun(paper_id=request.paper_id)
        
        if "status" in result and result["status"] == "error":
            logger.error(f"Download error: {result['message']}")
            raise HTTPException(status_code=500, detail=result["message"])
        
        logger.info(f"Paper downloaded successfully: {request.paper_id}")
        if "file_path" in result:
            logger.info(f"Downloaded to: {result['file_path']}")
            
        return result
    except Exception as e:
        error_msg = log_exception(logger, f"Error in download_paper endpoint for paper ID: {request.paper_id}", e)
        raise HTTPException(status_code=500, detail=error_msg)


# Main entry point
if __name__ == "__main__":
    import uvicorn
    from ..utils.config import API_HOST, API_PORT

    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=True)
