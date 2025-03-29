from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
import logging
import json
import os

from .search_agent import SearchAgent
from .summarization_agent import SummarizationAgent
from ..utils.config import GROQ_API_KEY, DEFAULT_MODEL, SAMBANOVA_API_KEY, SAMBANOVA_MODEL
from ..utils.debug_wrappers import DebugChatGroq, DebugChatSambanova

# Configure logging
logger = logging.getLogger("agent.orchestrator")


class OrchestratorAgentInput(BaseModel):
    """Input for the orchestrator agent."""

    query: str = Field(..., description="The user query")


class OrchestratorAgent:
    """
    Orchestrator agent responsible for directing queries to the appropriate agent.
    """

    def __init__(self, use_sambanova=False):
        self.search_agent = SearchAgent()
        self.summarization_agent = SummarizationAgent()
        
        # Choose which model to use based on the parameter
        if use_sambanova:
            # Use Sambanova model
            logger.info(f"Initializing SambanovaAPI with model: {SAMBANOVA_MODEL}, API Key: {SAMBANOVA_API_KEY[:4]}...{SAMBANOVA_API_KEY[-4:] if SAMBANOVA_API_KEY else 'None'}")
            self.llm = DebugChatSambanova(
                sambanova_api_key=SAMBANOVA_API_KEY, model_name=SAMBANOVA_MODEL, temperature=0
            )
        else:
            # Use our debug wrapper instead of regular ChatGroq
            logger.info(f"Initializing GroqAPI with model: {DEFAULT_MODEL}, API Key: {GROQ_API_KEY[:4]}...{GROQ_API_KEY[-4:] if GROQ_API_KEY else 'None'}")
            self.llm = DebugChatGroq(
                api_key=GROQ_API_KEY, model_name=DEFAULT_MODEL, temperature=0
            )
        
        self.tools = [
            Tool(
                name="search_for_papers",
                func=self._search_for_papers,
                description="Search for relevant papers based on a query",
            ),
            Tool(
                name="summarize_paper",
                func=self._summarize_paper,
                description="Summarize a paper given its ID",
            ),
        ]
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.agent_executor = self._create_agent_executor()

    async def _search_for_papers(self, query: str) -> Dict[str, Any]:
        """Forward the query to the search agent."""
        logger.info(f"Searching for papers with query: {query}")
        return await self.search_agent.process(query)

    async def _summarize_paper(self, paper_id: str) -> Dict[str, Any]:
        """Forward the request to the summarization agent."""
        logger.info(f"Summarizing paper with ID: {paper_id}")
        return await self.summarization_agent.process(paper_id)

    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor."""
        system_message = """You are a research assistant that helps users find and understand academic papers.
        
        Based on the user's query, determine if they want to:
        1. Search for relevant papers on a topic
        2. Get a summary of a specific paper
        
        For searches, use the search_for_papers tool with the user's query.
        For summaries, extract the paper ID from the query and use the summarize_paper tool.
        
        If the user mentions a paper by name but doesn't provide an ID, first search for the paper
        and then get its ID to summarize it.
        
        Respond in a helpful, concise manner.
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        logger.debug(f"Creating agent with prompt: {prompt}")
        agent = OpenAIFunctionsAgent(llm=self.llm, tools=self.tools, prompt=prompt)

        return AgentExecutor(
            agent=agent, tools=self.tools, memory=self.memory, verbose=True
        )

    async def process(self, query: str) -> Dict[str, Any]:
        """
        Process a user query.

        Args:
            query: The user query

        Returns:
            Results of processing the query
        """
        logger.debug(f"Processing query: {query}")
        try:
            result = await self.agent_executor.arun(input=query)
            logger.debug(f"Query result: {result}")
            return {"result": result, "status": "success"}
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {"error": str(e), "status": "error"}
