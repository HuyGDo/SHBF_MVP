import os
import glob
import uuid
import requests # Ensure 'requests' is installed: pip install requests
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
# Consider updating this import path if using newer langchain versions
from langchain.text_splitter import RecursiveCharacterTextSplitter
# OpenAIEmbeddings will only be used for its class structure if needed,
# but actual embedding calls will be direct.
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient, models
from typing import Union, List # Import Union and List for Python 3.9 compatibility

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "policy_documents_mvp")
EMBEDDINGS_API_URL = os.getenv("EMBEDDING_MODEL_API_URL", "http://localhost:1234/v1") # Ensure this is correct port
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "AITeamVN-Vietnamese_Embedding-gguf")

POLICY_DOCS_PATH = os.path.join(os.path.dirname(__file__), '..', "data_mvp", "policy_documents")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_documents_from_path(docs_path: str) -> List[TextLoader]: # Adjusted type hint
    """Loads all .txt documents from the specified directory."""
    print(f"Attempting to load documents from: {docs_path}")
    doc_files = glob.glob(os.path.join(docs_path, "*.txt"))
    if not doc_files:
        print(f"No .txt files found in {docs_path}")
    loaded_documents = []
    for doc_file in doc_files:
        try:
            loader = TextLoader(doc_file, encoding='utf-8')
            loaded_documents.extend(loader.load()) # loader.load() returns List[Document]
            print(f"Successfully loaded: {doc_file}")
        except Exception as e:
            print(f"Error loading {doc_file}: {e}")
    return loaded_documents

def chunk_documents(documents: list, chunk_size: int, chunk_overlap: int) -> list: # documents is List[Document]
    """Splits loaded documents into smaller chunks."""
    if not documents:
        print("No documents to chunk.")
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunked_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunked_docs)} chunks.")
    return chunked_docs

def _get_embedding_via_direct_request(text_to_embed: str, api_url: str, model_name: str) -> Union[List[float], None]: # Changed type hint
    """
    Helper function to get embedding for a single text via direct HTTP request.
    Returns the embedding vector or None if an error occurs.
    """
    full_embeddings_url = api_url.rstrip('/') + "/embeddings"
    payload = {"input": text_to_embed, "model": model_name}
    headers = {"Content-Type": "application/json"}

    try:
        # print(f"    Directly embedding text (first 30 chars): '{text_to_embed[:30]}...' via {full_embeddings_url}")
        # print(f"    Payload: {json.dumps(payload)}")
        response = requests.post(full_embeddings_url, headers=headers, json=payload, timeout=60) # Added timeout

        if response.status_code == 200:
            response_data = response.json()
            if "data" in response_data and response_data["data"] and "embedding" in response_data["data"][0]:
                embedding_vector = response_data["data"][0]["embedding"]
                if embedding_vector:
                    return embedding_vector
                else:
                    print(f"    Warning: Received empty embedding vector from direct request for text: '{text_to_embed[:30]}...'")
                    return None
            else:
                print(f"    Warning: Direct request successful (200 OK) but response format unexpected for text: '{text_to_embed[:30]}...'. Response: {response.text[:200]}")
                return None
        else:
            print(f"    Warning: Direct embedding request failed for text: '{text_to_embed[:30]}...' with status {response.status_code}. Response: {response.text[:200]}")
            return None
    except requests.exceptions.Timeout:
        print(f"    Warning: Timeout during direct embedding request for text: '{text_to_embed[:30]}...'")
        return None
    except Exception as e:
        print(f"    Warning: Exception during direct embedding for text: '{text_to_embed[:30]}...'. Error: {e}")
        return None

def ingest_data_to_qdrant(
    qdrant_client: QdrantClient,
    collection_name: str,
    document_chunks: list
    # embeddings_model parameter is no longer used for generation
):
    """
    Ingests document chunks and their embeddings into the specified Qdrant collection.
    Determines embedding dimension and creates collection if it doesn't exist.
    Uses direct HTTP requests for ALL embedding generations.
    """
    if not document_chunks:
        print("No document chunks to ingest.")
        return

    # 1. Determine embedding dimension dynamically using a direct HTTP request (as before)
    vector_size = None
    full_embeddings_url_for_check = "" # For error message context
    try:
        print("Determining embedding dimension using a direct HTTP request...")
        sample_text_for_dim_check = document_chunks[0].page_content[:100].strip()
        if not sample_text_for_dim_check: sample_text_for_dim_check = document_chunks[0].page_content[:500].strip()
        if not sample_text_for_dim_check: sample_text_for_dim_check = "sample text"
        print(f"Using sample text for dimension check (first 50 chars): '{sample_text_for_dim_check[:50]}...'")

        payload = {"input": sample_text_for_dim_check, "model": EMBEDDING_MODEL_NAME}
        headers = {"Content-Type": "application/json"}
        full_embeddings_url_for_check = EMBEDDINGS_API_URL.rstrip('/') + "/embeddings"
        
        print(f"Sending direct POST request for dimension check to: {full_embeddings_url_for_check}")
        print(f"Payload for dimension check: {json.dumps(payload)}")
        response = requests.post(full_embeddings_url_for_check, headers=headers, json=payload, timeout=60)
        
        print(f"Direct request (dim check) response status code: {response.status_code}")
        try:
            response_json_dim_check = response.json()
            print(f"Direct request (dim check) response JSON: {json.dumps(response_json_dim_check, indent=2)}")
        except json.JSONDecodeError:
            print(f"Direct request (dim check) response text (not JSON): {response.text}")

        if response.status_code == 200:
            response_data = response.json()
            if "data" in response_data and response_data["data"] and "embedding" in response_data["data"][0]:
                sample_embedding_vector = response_data["data"][0]["embedding"]
                if not sample_embedding_vector: raise ValueError("Empty embedding vector in dim check response.")
                vector_size = len(sample_embedding_vector)
                print(f"Detected embedding dimension via direct request: {vector_size}")
            else:
                error_detail = response_data.get("error", {}).get("message", str(response_data))
                raise ValueError(f"Dim check direct request successful (200 OK) but response format unexpected. Detail: {error_detail}")
        else:
            error_message = f"Dim check direct request failed with status {response.status_code}."
            try: error_detail = response.json().get("error", {}).get("message", response.text); error_message += f" Detail: {error_detail}"
            except json.JSONDecodeError: error_message += f" Response text: {response.text}"
            raise ValueError(error_message)
    except Exception as e:
        print(f"CRITICAL: Error determining embedding dimension with direct HTTP request: {e}")
        print(f"Endpoint attempted: {full_embeddings_url_for_check if full_embeddings_url_for_check else EMBEDDINGS_API_URL}")
        print("Cannot proceed without embedding dimension. Check LM Studio console and settings.")
        return

    # 2. Create collection if it doesn't exist
    try:
        qdrant_client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except Exception: # More specific exception handling could be qdrant_client.http.exceptions.UnexpectedResponse
        print(f"Collection '{collection_name}' not found. Creating new collection...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        )
        print(f"Collection '{collection_name}' created with vector size {vector_size}.")

    # 3. Prepare points for upsertion by embedding each chunk individually using direct HTTP requests
    points_to_upsert = []
    print(f"\nGenerating embeddings for {len(document_chunks)} chunks using DIRECT HTTP requests (one by one)...")

    for i, chunk in enumerate(document_chunks):
        if (i + 1) % 5 == 0 or i == 0 or (i+1) == len(document_chunks) : # Log progress more frequently for direct calls
             print(f"  Processing chunk {i+1}/{len(document_chunks)} from source: {chunk.metadata.get('source', 'Unknown')}")
        
        text_to_embed = chunk.page_content
        if not text_to_embed.strip():
            print(f"    Skipping empty chunk {i+1}.")
            continue

        # Use the helper function for direct embedding request
        vector = _get_embedding_via_direct_request(text_to_embed, EMBEDDINGS_API_URL, EMBEDDING_MODEL_NAME)

        if vector:
            point_id = str(uuid.uuid4())
            payload = {"text": chunk.page_content, "source": chunk.metadata.get("source", "Unknown")}
            points_to_upsert.append(models.PointStruct(id=point_id, vector=vector, payload=payload))
        else:
            # Warning already printed by _get_embedding_via_direct_request
            print(f"    Failed to embed chunk {i+1}. Skipping.")


    if not points_to_upsert:
        print("\nNo points were successfully prepared for upsertion after embedding. Check embedding errors above.")
        return
        
    print(f"\nSuccessfully generated embeddings for {len(points_to_upsert)} chunks via direct HTTP requests.")

    # 4. Upsert points to Qdrant
    print(f"Upserting {len(points_to_upsert)} points to Qdrant collection '{collection_name}'...")
    batch_size = 100 
    for i in range(0, len(points_to_upsert), batch_size):
        batch = points_to_upsert[i:i + batch_size]
        try:
            qdrant_client.upsert(collection_name=collection_name, points=batch, wait=True)
            print(f"  Upserted batch {i // batch_size + 1}/{(len(points_to_upsert) -1) // batch_size + 1} (size: {len(batch)})")
        except Exception as e_upsert:
            print(f"  Error upserting batch {i // batch_size + 1}: {e_upsert}")
    print("All prepared points attempted for upsertion.")

def main():
    """Main function to run the data ingestion pipeline."""
    print("--- Starting Data Ingestion Process ---")

    try:
        qdrant_client = QdrantClient(url=QDRANT_URL)
        qdrant_client.get_collections() 
        print(f"Successfully connected to Qdrant at {QDRANT_URL}")
    except Exception as e:
        print(f"Failed to connect to Qdrant at {QDRANT_URL}: {e}. Please ensure Qdrant is running.")
        return

    documents = load_documents_from_path(POLICY_DOCS_PATH)
    if not documents: print("No documents found or loaded. Exiting."); return

    chunked_docs = chunk_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunked_docs: print("No chunks created from documents. Exiting."); return
        
    # The OpenAIEmbeddings object is no longer strictly needed for generation
    # as we are bypassing its .embed_query() method for direct HTTP calls.
    # We don't need to call get_embeddings_model() here anymore.
    # try:
    #     _ = get_embeddings_model(EMBEDDINGS_API_URL, EMBEDDING_MODEL_NAME) # Still call for consistency of setup logging
    # except Exception as e:
    #     print(f"Note: Failed to initialize OpenAIEmbeddings object (not critical as using direct HTTP): {e}")

    # --- Added code to write chunks to file ---
    output_chunks_path = os.path.join(POLICY_DOCS_PATH, "chunked.txt")
    try:
        with open(output_chunks_path, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunked_docs):
                f.write(chunk.page_content)
                if i < len(chunked_docs) - 1:
                    f.write("\n\n---\n\n") # Separator
        print(f"Successfully wrote {len(chunked_docs)} chunks to {output_chunks_path}")
    except Exception as e:
        print(f"Error writing chunks to file {output_chunks_path}: {e}")
    # --- End of added code ---

    try:
        ingest_data_to_qdrant(qdrant_client, QDRANT_COLLECTION_NAME, chunked_docs)
    except Exception as e:
        print(f"An error occurred during the data ingestion main workflow: {e}")
        
    print("--- Data Ingestion Process Finished ---")

if __name__ == "__main__":
    main()
