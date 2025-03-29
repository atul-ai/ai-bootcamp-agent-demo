from langchain_groq import ChatGroq
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("groq_requests.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("groq_agent")


class DebugChatGroq(ChatGroq):
    """Wrapper around ChatGroq that logs requests and responses."""
    
    def _create_message_dicts(self, messages, stop=None):
        """Log the messages before sending to the API.
        
        Args:
            messages: The messages to send to the API
            stop: Stop sequences
        
        Returns:
            The message dicts and params
        """
        message_dicts, params = super()._create_message_dicts(messages, stop)
        logger.info(f"Sending to Groq API (model={self.model_name}):\n{json.dumps(message_dicts, indent=2, default=str)}")
        return message_dicts, params
    
    def _create_chat_result(self, response):
        """Log the API response."""
        # Convert to dict with default serializer for non-serializable objects
        try:
            logger.info(f"Received from Groq API:\n{json.dumps(response.dict(), indent=2, default=str)}")
        except Exception as e:
            logger.warning(f"Could not serialize response: {e}")
            logger.info(f"Received raw response from Groq API")
        
        return super()._create_chat_result(response)


class DebugChatSambanova(ChatSambaNovaCloud):
    """Wrapper around ChatSambaNovaCloud that logs requests and responses."""
    
    def _create_message_dicts(self, messages, stop=None):
        """Log the messages before sending to the API.
        
        Args:
            messages: The messages to send to the API
            stop: Stop sequences
        
        Returns:
            The message dicts and params
        """
        message_dicts, params = super()._create_message_dicts(messages, stop)
        logger.info(f"Sending to Sambanova API (model={self.model_name}):\n{json.dumps(message_dicts, indent=2, default=str)}")
        return message_dicts, params
    
    def _create_chat_result(self, response):
        """Log the API response."""
        # Convert to dict with default serializer for non-serializable objects
        try:
            logger.info(f"Received from Sambanova API:\n{json.dumps(response.dict(), indent=2, default=str)}")
        except Exception as e:
            logger.warning(f"Could not serialize response: {e}")
            logger.info(f"Received raw response from Sambanova API")
        
        return super()._create_chat_result(response) 