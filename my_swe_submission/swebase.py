import os
import requests
from coding.schemas import Patch, Edit
from abc import ABC, abstractmethod


# if host ip is localhost itll fail, need to get docker host ip
class LLMClient:
    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        auth_key: str = None,
    ):
        """Initialize LLM client with API server URL"""
        # Get values from environment if not provided
        self.base_url = (base_url or f"http://{os.getenv('HOST_IP', 'localhost')}:25000").rstrip("/")
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.auth_key = auth_key or os.getenv("LLM_AUTH_KEY", "")
        
        # Initialize the API key with the LLM service
        self._init_key()

    def _init_key(self):
        """Initialize the API key with the LLM service"""
        if not self.api_key:
            print("⚠️ No OPENROUTER_API_KEY provided, skipping initialization")
            return
        
        if not self.auth_key:
            print("⚠️ No LLM_AUTH_KEY provided, skipping initialization")
            return
            
        try:
            print(f"🔑 Initializing API key with LLM service at {self.base_url}")
            
            # Prepare the initialization payload
            init_payload = {"key": self.api_key}
            
            # Add auth key as header (same format as manager.py)
            headers = {"Authorization": self.auth_key}
            
            response = requests.post(
                f"{self.base_url}/init", 
                json=init_payload,
                headers=headers
            )
            response.raise_for_status()
            
            result = response.json()
            print(f"✅ LLM service initialized: {result.get('message', 'Success')}")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to initialize LLM service: {e}")
            # Don't raise here - let the system continue and handle errors during actual calls

    def __call__(
        self,
        query: str,
        llm_name: str,
        temperature: float = 0.7,
        max_tokens: int = 16384,
    ) -> tuple[str, int]:
        """
        Call LLM API endpoint

        Args:
            query (str): The prompt/query to send to the LLM
            llm_name (str): Name of LLM model to use (e.g. "gpt-4", "claude-3-sonnet")
            temperature (float): Temperature for the LLM
        Returns:
            tuple[str, int]: (Generated response text, Total tokens used for this key)

        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        payload = {
            "query": query,
            "llm_name": llm_name,
            "temperature": temperature,
            "api_key": self.api_key,
            "max_tokens": max_tokens,
        }

        response = requests.post(f"{self.base_url}/call", json=payload)
        response.raise_for_status()

        result = response.json()
        return result["result"], result["total_tokens"]

    def embed(self, query: str) -> list[float]:
        """
        Get embeddings for text using the embedding API endpoint

        Args:
            query (str): The text to get embeddings for

        Returns:
            list[float]: Vector embedding of the input text

        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        payload = {"query": query}

        response = requests.post(f"{self.base_url}/embed", json=payload)
        response.raise_for_status()

        result = response.json()
        return result["vector"]

    def embed_documents(self, queries: list[str]) -> list[list[float]]:
        """
        Get embeddings for text using the embedding API endpoint

        Args:
            queries (list[str]): The list of texts to get embeddings for

        Returns:
            list[list[float]]: Vector embedding of the input text

        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        payload = {"queries": queries}

        response = requests.post(f"{self.base_url}/embed/batch", json=payload)
        response.raise_for_status()

        result = response.json()
        return result["vectors"]


class SWEBase(ABC):
    def __init__(self):
        self.llm = LLMClient()

    @abstractmethod
    def __call__(self, repo_location: str, issue_description: str) -> Patch:
        pass
