from typing import Dict, Any, List
from langchain.tools import BaseTool
import httpx
import json
import os

from ..utils.config import GROQ_API_KEY, DEFAULT_MODEL


class PaperSummarizerTool(BaseTool):
    name: str = "paper_summarizer"
    description: str = "Generate a summary of a paper using an LLM"

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        super().__init__()
        self.model_name = model_name
        self.api_key = GROQ_API_KEY
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    def _run(
        self, paper_content: str, paper_title: str = None, max_tokens: int = 1000
    ) -> str:
        """
        Generate a summary of a paper.

        Args:
            paper_content: The content of the paper
            paper_title: The title of the paper (optional)
            max_tokens: Maximum number of tokens in the summary

        Returns:
            A summary of the paper
        """
        # Truncate content if too long (simple approach)
        if len(paper_content) > 20000:
            paper_content = paper_content[:20000] + "..."

        # Prepare the prompt
        title_info = f"Title: {paper_title}\n\n" if paper_title else ""
        system_prompt = """You are an academic research assistant. Your task is to summarize the 
        provided academic paper in a concise but informative manner. Include:
        1. Main research question/objective
        2. Methodology
        3. Key findings
        4. Implications/conclusions
        Make the summary accessible to researchers who want to quickly understand the paper's importance."""

        user_prompt = f"{title_info}Paper content:\n{paper_content}\n\nPlease summarize this paper."

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1,
            }

            response = httpx.post(
                self.api_url, headers=headers, json=payload, timeout=60.0
            )

            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return f"Error generating summary: {result.get('error', {}).get('message', 'Unknown error')}"

        except Exception as e:
            return f"Error generating summary: {str(e)}"
