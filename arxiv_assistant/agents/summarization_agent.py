from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
import logging

from ..tools.paper_downloader import ArxivPaperDownloaderTool
from ..tools.summarizer import PaperSummarizerTool
from ..utils.config import GROQ_API_KEY, DEFAULT_MODEL
from ..utils.debug_wrappers import DebugChatGroq

# Get logger
logger = logging.getLogger("agent.summarization")


class SummarizationAgentInput(BaseModel):
    """Input for the summarization agent."""

    paper_id: str = Field(..., description="The ID of the paper to summarize")


class SummarizationAgent:
    """Agent responsible for summarizing papers."""

    def __init__(self):
        self.paper_downloader = ArxivPaperDownloaderTool()
        self.paper_summarizer = PaperSummarizerTool()
        # Use the debug wrapper for ChatGroq
        logger.info(f"Initializing Summarization Agent with model: {DEFAULT_MODEL}")
        self.llm = DebugChatGroq(
            api_key=GROQ_API_KEY, model_name=DEFAULT_MODEL, temperature=0
        )
        self.tools = [
            Tool(
                name="download_paper",
                func=self.paper_downloader._run,
                description="Download a paper from arXiv using its ID",
            ),
            Tool(
                name="summarize_paper",
                func=self.paper_summarizer._run,
                description="Generate a summary of a paper using an LLM",
            ),
        ]
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.agent_executor = self._create_agent_executor()

    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor."""
        system_message = """You are a research paper summarization assistant.
        Your job is to help users get summaries of academic papers.
        
        First, use the download_paper tool to get the paper content.
        Then, use the summarize_paper tool to generate a summary.
        
        Present the summary in a clear, organized format with key findings highlighted.
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        logger.debug(f"Creating summarization agent with prompt: {prompt}")
        agent = OpenAIFunctionsAgent(llm=self.llm, tools=self.tools, prompt=prompt)

        return AgentExecutor(
            agent=agent, tools=self.tools, memory=self.memory, verbose=True
        )

    async def process(self, paper_id: str) -> Dict[str, Any]:
        """
        Process a paper summarization request.

        Args:
            paper_id: The ID of the paper to summarize

        Returns:
            Results of the summarization
        """
        logger.info(f"Processing summarization request for paper ID: {paper_id}")
        try:
            result = await self.agent_executor.arun(
                input=f"Summarize the paper with ID {paper_id}"
            )
            logger.debug(f"Summarization result: {result[:100] if result else ''}...")
            return {"result": result, "status": "success"}
        except Exception as e:
            logger.error(f"Error processing summarization request: {str(e)}", exc_info=True)
            return {"error": str(e), "status": "error"}
