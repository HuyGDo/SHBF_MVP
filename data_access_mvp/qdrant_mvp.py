from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, SearchRequest, Filter
from typing import List, Dict, Any, Optional
import logging
from config_mvp.settings_mvp import QDRANT_URL, QDRANT_COLLECTION_NAME
from core_ai_mvp.embeddings.embedding_client_mvp import get_embedding_client_mvp # Assuming this provides embed_query

logger = logging.getLogger(__name__)

def qdrant_similarity_search(
    query_text: Optional[str] = None,
    query_vector: Optional[List[float]] = None,
    collection_name: str = QDRANT_COLLECTION_NAME,
    qdrant_url: str = QDRANT_URL,
    top_k: int = 3,
    score_threshold: Optional[float] = None,
    search_filter: Optional[Filter] = None
) -> List[Dict[str, Any]]:
    """
    Connects to Qdrant and performs a similarity search using either a query text (which will be embedded)
    or a pre-computed query vector.

    Args:
        query_text (Optional[str]): The text to search for. Will be embedded using the embedding client.
                                    Either query_text or query_vector must be provided.
        query_vector (Optional[List[float]]): The vector to search with.
                                             Either query_text or query_vector must be provided.
        collection_name (str): The name of the Qdrant collection to search in.
                               Defaults to QDRANT_COLLECTION_NAME from settings.
        qdrant_url (str): The URL of the Qdrant instance. Defaults to QDRANT_URL from settings.
        top_k (int): The number of top results to return. Defaults to 3.
        score_threshold (Optional[float]): Minimum similarity score for a result to be included.
                                           Defaults to None (no threshold).
        search_filter (Optional[Filter]): Qdrant filter conditions to apply during search.
                                          Defaults to None (no filter).

    Returns:
        List[Dict[str, Any]]: A list of search results (scored points), where each result is a dictionary
                              containing 'id', 'score', 'payload', and potentially 'vector'.
                              Returns an empty list if no results or in case of an error.
    Raises:
        ValueError: If neither query_text nor query_vector is provided.
    """
    if not query_text and not query_vector:
        raise ValueError("Either query_text or query_vector must be provided for similarity search.")

    try:
        logger.info(f"Connecting to Qdrant at {qdrant_url} and accessing collection '{collection_name}'.")
        client = QdrantClient(url=qdrant_url)
        client.get_collection(collection_name=collection_name) # Check if collection exists

        if query_text and not query_vector:
            logger.info("Query text provided. Getting embedding client to generate vector...")
            embedding_model = get_embedding_client_mvp()
            logger.info(f"Embedding query: '{query_text[:80]}...'") # Log snippet of query
            query_vector = embedding_model.embed_query(query_text)
            logger.info(f"Successfully generated query vector with dimension {len(query_vector)}.")

        logger.info(f"Performing search in Qdrant with top_k={top_k}...")
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=search_filter, 
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True # Ensure payload is returned
        )
        logger.info(f"Qdrant search completed. Found {len(search_result)} results.")
        
        # Convert ScoredPoint objects to dictionaries for easier use
        results = [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
                # "vector": hit.vector # Optionally include the vector
            }
            for hit in search_result
        ]
        return results

    except Exception as e:
        logger.error(f"Error during Qdrant similarity search in collection '{collection_name}': {e}", exc_info=True)
        # In a real application, you might want to handle different exceptions more granularly
        # or re-raise them after logging.
        return [] # Return empty list on error for MVP simplicity

if __name__ == '__main__':
    print(f"Attempting to connect to Qdrant at: {QDRANT_URL}")
    print(f"Target collection: {QDRANT_COLLECTION_NAME}")
    print("Ensure your Qdrant instance is running and the collection exists and has data.")
    print("Also, ensure your embedding model server (e.g., LM Studio) is running for text-based queries.")

    # Example Usage
    # You would need to have a Qdrant collection populated with vectors.
    # For this example, we'll try a text query. It relies on the embedding client working.

    sample_query_text = "What are the conditions for a personal loan?"
    print(f"\n--- Testing with text query: '{sample_query_text}' ---")
    
    try:
        # This test requires the embedding model to be available via get_embedding_client_mvp()
        # and the Qdrant collection to be populated with compatible vectors.
        search_results = qdrant_similarity_search(query_text=sample_query_text)
        
        if search_results:
            print(f"Found {len(search_results)} results:")
            for i, hit in enumerate(search_results):
                print(f"  Result {i+1}:")
                print(f"    ID: {hit['id']}")
                print(f"    Score: {hit['score']:.4f}")
                print(f"    Payload: {hit['payload']}")
        else:
            print("No results found or an error occurred during the search.")
            print("Possible reasons:")
            print("  - Qdrant server not running or collection doesn't exist.")
            print("  - Embedding model server not running or model not loaded.")
            print("  - Collection is empty or does not contain relevant vectors.")

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during the Qdrant example: {e}")
        print("This could be an issue with Qdrant connection, embedding model, or the collection itself.")

    # Example with a dummy vector (if you don't want to test embedding generation)
    # This requires knowing the dimension of vectors in your QDRANT_COLLECTION_NAME
    # For example, if your vectors are 384-dimensional:
    # DUMMY_VECTOR_DIM = 384 
    # dummy_vector = [0.1] * DUMMY_VECTOR_DIM 
    # print(f"\n--- Testing with a dummy vector (dim={DUMMY_VECTOR_DIM}) ---")
    # try:
    #     search_results_vector = qdrant_similarity_search(query_vector=dummy_vector, top_k=2)
    #     if search_results_vector:
    #         print(f"Found {len(search_results_vector)} results (with dummy vector):")
    #         for hit in search_results_vector:
    #             print(f"    ID: {hit['id']}, Score: {hit['score']:.4f}, Payload: {hit['payload']}")
    #     else:
    #         print("No results found with dummy vector or an error occurred.")
    # except Exception as e:
    #     print(f"Error with dummy vector search: {e}")

    print("\nTo run these tests effectively, ensure:")
    print("1. Qdrant is running and the specified collection exists with data.")
    print("2. Your embedding model (e.g., in LM Studio) is running for text queries.")
    print("3. `core_ai_mvp/embeddings/embedding_client_mvp.py` is correctly configured.")
