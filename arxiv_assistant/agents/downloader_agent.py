# import os
# # Set tokenizers parallelism to avoid deadlocks with forked processes
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# import arxiv
# import logging
# import tempfile
# from typing import Dict, Any, Optional
# import aiofiles
# import asyncio
# from pathlib import Path

# from ..utils.config import DATA_DIR, TEMP_DIR

# # Set up logging
# logger = logging.getLogger(__name__)

# class DownloaderAgent:
#     """
#     Agent responsible for downloading papers from arXiv.
#     """

#     def __init__(self, download_dir: Optional[str] = None):
#         """
#         Initialize the downloader agent.

#         Args:
#             download_dir: Directory to store downloaded papers.
#                           If None, uses the configured data directory.
#         """
#         self.download_dir = download_dir or os.path.join(DATA_DIR, "papers")
#         # Create the download directory if it doesn't exist
#         os.makedirs(self.download_dir, exist_ok=True)
#         logger.info(f"DownloaderAgent initialized with download directory: {self.download_dir}")
        
#         # Create temp dir if not exists
#         os.makedirs(TEMP_DIR, exist_ok=True)

#     async def download_paper(self, paper_id: str) -> Dict[str, Any]:
#         """
#         Download a paper from arXiv by its ID.

#         Args:
#             paper_id: The arXiv ID of the paper to download

#         Returns:
#             A dictionary containing the paper information and local file path
#             or an error message
#         """
#         try:
#             logger.info(f"Downloading paper with ID: {paper_id}")
            
#             # Clean the paper ID (remove version if present)
#             clean_id = paper_id.split('v')[0] if 'v' in paper_id else paper_id
            
#             # Define the output path
#             output_path = os.path.join(self.download_dir, f"{clean_id}.pdf")
            
#             # Check if the paper is already downloaded
#             if os.path.exists(output_path):
#                 logger.info(f"Paper {paper_id} already downloaded: {output_path}")
#                 return {
#                     "status": "success",
#                     "message": f"Paper already downloaded",
#                     "paper_id": paper_id,
#                     "file_path": output_path
#                 }
            
#             # Use arxiv client to download the paper
#             # Since arxiv API is synchronous, run it in a separate thread
#             client = arxiv.Client()
            
#             # Create a search query for the paper ID
#             search = arxiv.Search(id_list=[clean_id])
            
#             loop = asyncio.get_event_loop()
#             result = await loop.run_in_executor(None, lambda: list(client.results(search)))
            
#             if not result:
#                 error_msg = f"Paper with ID {paper_id} not found on arXiv"
#                 logger.error(error_msg)
#                 return {"status": "error", "message": error_msg}
            
#             paper = result[0]
            
#             # Download the paper
#             temp_file = await loop.run_in_executor(None, lambda: paper.download_pdf(dirpath=tempfile.gettempdir()))
            
#             # Move the file to the final location
#             async with aiofiles.open(temp_file, 'rb') as src_file:
#                 content = await src_file.read()
                
#             async with aiofiles.open(output_path, 'wb') as dest_file:
#                 await dest_file.write(content)
            
#             # Remove the temporary file
#             os.remove(temp_file)
            
#             logger.info(f"Successfully downloaded paper {paper_id} to {output_path}")
            
#             # Return the paper information
#             return {
#                 "status": "success",
#                 "message": "Paper downloaded successfully",
#                 "paper_id": paper_id,
#                 "file_path": output_path,
#                 "title": paper.title,
#                 "authors": ", ".join([author.name for author in paper.authors]),
#                 "summary": paper.summary,
#                 "published": str(paper.published),
#                 "updated": str(paper.updated),
#                 "pdf_url": paper.pdf_url
#             }
            
#         except Exception as e:
#             error_msg = f"Error downloading paper {paper_id}: {str(e)}"
#             logger.error(error_msg, exc_info=True)
#             return {"status": "error", "message": error_msg} 