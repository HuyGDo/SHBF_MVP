import requests
from typing import List, Optional
import json
from config_mvp.settings_mvp import EMBEDDING_MODEL_API_URL, EMBEDDING_MODEL_NAME

class EmbeddingClientMvp:
    def __init__(self, api_url: str = EMBEDDING_MODEL_API_URL, model_name: str = EMBEDDING_MODEL_NAME):
        self.api_url = api_url.rstrip('/')
        self.model_name = model_name
        self.headers = {"Content-Type": "application/json"}

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single text using direct HTTP request to LM Studio."""
        if not text.strip():
            raise ValueError("Cannot embed empty text")

        payload = {
            "input": text.strip(),
            "model": self.model_name
        }

        try:
            response = requests.post(
                f"{self.api_url}/embeddings",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            if "data" in result and result["data"] and "embedding" in result["data"][0]:
                return result["data"][0]["embedding"]
            else:
                raise ValueError(f"Unexpected response format: {result}")
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds multiple texts using direct HTTP requests to LM Studio."""
        if not texts:
            return []
        
        results = []
        for text in texts:
            results.append(self.embed_query(text))
        return results

def get_embedding_client_mvp() -> EmbeddingClientMvp:
    """
    Returns an instance of the embedding client configured for LM Studio.
    """
    return EmbeddingClientMvp()

if __name__ == '__main__':
    print(f"Testing Embedding Client MVP with:")
    print(f"  API URL: {EMBEDDING_MODEL_API_URL}")
    print(f"  Model Name: {EMBEDDING_MODEL_NAME}")

    try:
        client = get_embedding_client_mvp()
        
        # Test single text embedding
        test_text = "Xin chào, đây là một câu ví dụ."
        print(f"\nTesting single text embedding: '{test_text}'")
        embedding = client.embed_query(test_text)
        print(f"Generated embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")

        # Test multiple texts embedding
        test_texts = ["Hello world!", "This is another test."]
        print(f"\nTesting multiple texts embedding: {test_texts}")
        embeddings = client.embed_documents(test_texts)
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Each embedding dimension: {len(embeddings[0])}")

    except Exception as e:
        print(f"\nError during embedding client test: {e}")
        print("\nPlease check:")
        print(f"1. LM Studio is running and accessible at {EMBEDDING_MODEL_API_URL}")
        print(f"2. The model '{EMBEDDING_MODEL_NAME}' is loaded")
        print("3. The server is properly configured for embeddings")
