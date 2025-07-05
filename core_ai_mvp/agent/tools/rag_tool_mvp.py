from langchain.tools import BaseTool
from typing import Type, Any, Dict, List, Optional
from pydantic import BaseModel, Field
from data_access_mvp.qdrant_mvp import qdrant_similarity_search, QDRANT_COLLECTION_NAME
from config_mvp.settings_mvp import QDRANT_URL # QDRANT_COLLECTION_NAME already imported
import logging # Added logging

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# Basic config will be inherited if this module is imported by another that sets it up.
# For standalone script execution, basicConfig might be needed in if __name__ == '__main__'.

class RagToolInput(BaseModel):
    policy_query_prompt: str = Field(description="The user query to search for in policy documents.")
    top_k: Optional[int] = Field(default=3, description="Number of top documents to retrieve.")
    score_threshold: Optional[float] = Field(default=None, description="Minimum similarity score for retrieved documents.")

class RagToolMvp(BaseTool):
    name: str = "policy_document_retriever"
    description: str = ("Useful for retrieving relevant policy document snippets based on a user's query. "
                        "Input should be the user's query about policies.")
    args_schema: Type[BaseModel] = RagToolInput

    def _run(self, policy_query_prompt: str, top_k: Optional[int] = 3, score_threshold: Optional[float] = None, **kwargs: Any) -> List[Dict[str, Any]]:
        """Executes the RAG tool.
        
        Internally calls qdrant_similarity_search which handles embedding the query_text.
        """
        if not policy_query_prompt:
            return [{"error": "No policy query prompt provided."}]
        
        # Clean the query text by removing extra whitespace and newlines
        policy_query_prompt = " ".join(policy_query_prompt.split())
        
        logger.info(f"Received policy query: '{policy_query_prompt}', top_k={top_k}, score_threshold={score_threshold}")

        try:
            # qdrant_similarity_search will use its default embedding client if only query_text is passed
            search_results = qdrant_similarity_search(
                query_text=policy_query_prompt,
                collection_name=QDRANT_COLLECTION_NAME, # from qdrant_mvp or settings
                qdrant_url=QDRANT_URL, # from settings
                top_k=top_k if top_k is not None else 3,
                score_threshold=score_threshold
            )
            
            if not search_results:
                return [{"info": "No relevant policy documents found."}]
            
            # For MVP, we can return the payload directly. 
            # Consider what parts of payload are most useful (e.g., text content, source)
            # Assuming payload contains a 'text_chunk' and 'source' field from ingestion script
            formatted_results = []
            for hit in search_results:
                content = hit.get('payload', {}).get('page_content', 'No content available')
                source = hit.get('payload', {}).get('source', 'Unknown source')
                score = hit.get('score')
                formatted_results.append({
                    "content": content,
                    "source": source,
                    "score": score
                })
            
            logger.info(f"Returning {len(formatted_results)} snippets.")
            # --- Added logging for retrieved chunks ---
            logger.info("Retrieved chunks from Qdrant:")
            for i, res_dict in enumerate(formatted_results):
                logger.info(f"  Chunk {i+1}:")
                logger.info(f"    Source: {res_dict.get('source')}")
                logger.info(f"    Score: {res_dict.get('score')}")
                logger.info(f"    Content: {res_dict.get('content')}") # Log full content
            # --- End of added logging ---
            return formatted_results
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return [{f"error": f"Failed to retrieve policy documents due to: {str(e)}"}]

    async def _arun(self, policy_query_prompt: str, **kwargs: Any) -> List[Dict[str, Any]]:
        # For MVP, a synchronous call is often sufficient.
        # If true async is needed, qdrant_similarity_search would need an async version.
        raise NotImplementedError("RagToolMvp does not support async execution yet.")

if __name__ == '__main__':
    # Setup basic logging for standalone script execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Testing RagToolMvp...")
    logger.info("Ensure Qdrant and the embedding model server (e.g., LM Studio) are running.")
    logger.info(f"Using Qdrant collection: {QDRANT_COLLECTION_NAME} at {QDRANT_URL}")

    rag_tool = RagToolMvp()

    
