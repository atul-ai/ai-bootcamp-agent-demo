from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
import logging

from ..tools.embedding_lookup import EmbeddingLookupTool
from ..utils.config import GROQ_API_KEY, DEFAULT_MODEL
from ..utils.debug_wrappers import DebugChatGroq

# Get logger
logger = logging.getLogger("agent.search")


class SearchAgentInput(BaseModel):
    """Input for the search agent."""

    query: str = Field(..., description="The search query for finding relevant papers")


class SearchAgent:
    """Agent responsible for searching for relevant papers."""

    def __init__(self):
        self.embedding_lookup_tool = EmbeddingLookupTool()
        # Use the debug wrapper for ChatGroq
        logger.info(f"Initializing Search Agent with model: {DEFAULT_MODEL}")
        self.llm = DebugChatGroq(
            api_key=GROQ_API_KEY, model_name=DEFAULT_MODEL, temperature=0
        )
        self.tools = [
            Tool(
                name="search_papers",
                func=self.embedding_lookup_tool._run,
                description="Search for academic papers related to a query",
            )
        ]
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.agent_executor = self._create_agent_executor()

    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor."""
        system_message = """You are a research paper search assistant. 
        Your job is to help users find relevant academic papers based on their queries.
        Use the search_papers tool to find papers relevant to the user's query.
        Analyze the results and provide a concise list of the most relevant papers with titles and brief descriptions.
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        logger.debug(f"Creating search agent with prompt: {prompt}")
        agent = OpenAIFunctionsAgent(llm=self.llm, tools=self.tools, prompt=prompt)

        return AgentExecutor(
            agent=agent, tools=self.tools, memory=self.memory, verbose=True
        )

    async def process(self, query: str) -> Dict[str, Any]:
        """
        Process a search query.

        Args:
            query: The search query

        Returns:
            Results of the search
        """
        logger.info(f"Processing search query: {query}")
        try:
            result = await self.agent_executor.arun(input=query)
            logger.debug(f"Search result: {result[:100] if result else ''}...")
            return {"result": result, "status": "success"}
        except Exception as e:
            logger.error(f"Error processing search query: {str(e)}", exc_info=True)
            return {"error": str(e), "status": "error"}
