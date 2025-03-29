from typing import Dict, Any, Optional, Awaitable
from langchain.tools import BaseTool
import arxiv
import os
import pypdf
import asyncio
import tempfile
import aiofiles
import logging

from ..utils.config import PAPER_CACHE_DIR

# Set up logging
logger = logging.getLogger(__name__)

class ArxivPaperDownloaderTool(BaseTool):
    name: str = "arxiv_paper_downloader"
    description: str = "Download papers from arXiv based on paper ID and extract their content"

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self):
        super().__init__()
        # Create paper cache directory if it doesn't exist
        os.makedirs(PAPER_CACHE_DIR, exist_ok=True)
        logger.info(f"Paper downloader initialized with cache directory: {PAPER_CACHE_DIR}")

    def _run(self, paper_id: str) -> Dict[str, Any]:
        """
        Synchronous interface for downloading a paper (runs the async method)
        
        Args:
            paper_id: The arXiv ID of the paper (e.g., '2104.08653')
            
        Returns:
            Dictionary with paper metadata and content
        """
        loop = asyncio.get_event_loop()
        try:
            if loop.is_running():
                # We're already in an async context
                # We need to use asyncio.create_task and a Future to get the result
                future = asyncio.Future()
                
                async def _set_future_result():
                    result = await self._arun(paper_id)
                    future.set_result(result)
                    
                asyncio.create_task(_set_future_result())
                return loop.run_until_complete(future)
            else:
                # No running loop, create one
                return loop.run_until_complete(self._arun(paper_id))
        except RuntimeError:
            # No event loop, so create one
            return asyncio.run(self._arun(paper_id))

    async def _arun(self, paper_id: str) -> Dict[str, Any]:
        """
        Download a paper from arXiv and extract its text content asynchronously.

        Args:
            paper_id: The arXiv ID of the paper (e.g., '2104.08653')

        Returns:
            Dictionary with paper metadata and content
        """
        try:
            logger.info(f"Downloading paper with ID: {paper_id}")
            
            # Clean the paper ID (remove version if present)
            if paper_id.startswith("arxiv:"):
                paper_id = paper_id[6:]
                
            clean_id = paper_id.split('v')[0] if 'v' in paper_id else paper_id
            
            # Define the output path
            output_path = os.path.join(PAPER_CACHE_DIR, f"{clean_id}.pdf")
            
            # Check if the paper is already downloaded
            if os.path.exists(output_path):
                logger.info(f"Paper {paper_id} already downloaded: {output_path}")
                
                # Extract text from the cached PDF
                paper_text = await self._extract_text_from_pdf(output_path)
                
                # Get paper metadata if possible
                try:
                    client = arxiv.Client()
                    search = arxiv.Search(id_list=[clean_id])
                    
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: list(client.results(search)))
                    
                    if result:
                        paper = result[0]
                        return {
                            "status": "success",
                            "message": f"Paper already downloaded",
                            "paper_id": paper_id,
                            "file_path": output_path,
                            "title": paper.title,
                            "authors": ", ".join([author.name for author in paper.authors]),
                            "abstract": paper.summary,
                            "published": str(paper.published),
                            "content": paper_text,
                            "cached": True
                        }
                    else:
                        return {
                            "status": "success",
                            "message": f"Paper already downloaded",
                            "paper_id": paper_id,
                            "file_path": output_path,
                            "content": paper_text,
                            "cached": True
                        }
                except Exception as e:
                    # If we can't get metadata, just return the text
                    return {
                        "status": "success",
                        "message": f"Paper already downloaded",
                        "paper_id": paper_id,
                        "file_path": output_path,
                        "content": paper_text,
                        "cached": True
                    }
            
            # Use arxiv client to download the paper
            # Since arxiv API is synchronous, run it in a separate thread
            client = arxiv.Client()
            
            # Create a search query for the paper ID
            search = arxiv.Search(id_list=[clean_id])
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: list(client.results(search)))
            
            if not result:
                error_msg = f"Paper with ID {paper_id} not found on arXiv"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            
            paper = result[0]
            
            # Download the paper
            temp_file = await loop.run_in_executor(None, lambda: paper.download_pdf(dirpath=tempfile.gettempdir()))
            
            # Move the file to the final location
            async with aiofiles.open(temp_file, 'rb') as src_file:
                content = await src_file.read()
                
            async with aiofiles.open(output_path, 'wb') as dest_file:
                await dest_file.write(content)
            
            # Remove the temporary file
            os.remove(temp_file)
            
            logger.info(f"Successfully downloaded paper {paper_id} to {output_path}")
            
            # Extract text from the PDF
            paper_text = await self._extract_text_from_pdf(output_path)
            
            # Return the paper information
            return {
                "status": "success",
                "message": "Paper downloaded successfully",
                "paper_id": paper_id,
                "file_path": output_path,
                "title": paper.title,
                "authors": ", ".join([author.name for author in paper.authors]),
                "abstract": paper.summary,
                "published": str(paper.published),
                "updated": str(paper.updated),
                "pdf_url": paper.pdf_url,
                "content": paper_text,
                "cached": False
            }
            
        except Exception as e:
            error_msg = f"Error downloading paper {paper_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"status": "error", "message": error_msg}
            
    async def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file asynchronously."""
        try:
            loop = asyncio.get_event_loop()
            
            def _extract():
                reader = pypdf.PdfReader(pdf_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
                
            text = await loop.run_in_executor(None, _extract)
            
            # Truncate if too long
            if len(text) > 30000:
                logger.info(f"Truncating PDF text from {len(text)} to 30000 characters")
                text = text[:30000] + "... [content truncated]"
                
            return text
        except Exception as e:
            error_msg = f"Error extracting text from PDF: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"Error extracting text: {str(e)}"
