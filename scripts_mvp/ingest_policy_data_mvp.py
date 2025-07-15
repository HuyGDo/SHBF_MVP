import os
import uuid
import requests
import json
import re
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.text_splitter import TokenTextSplitter
from qdrant_client import QdrantClient, models
from typing import Union, List

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "policy_documents_mvp")
EMBEDDINGS_API_URL = os.getenv("EMBEDDING_MODEL_API_URL", "http://10.2.20.2:1235/v1/embeddings")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-aiteamvn-vietnamese_embedding")

POLICY_DOCS_PATH = os.path.join(os.path.dirname(__file__), '..', "data_mvp", "policy_documents")
POLICY_FILENAME = "BẢN_ĐIỀU_KHOẢN_ĐIỀU_KIỆ̂N.txt"

# --- Chunking Strategy ---
# Pattern to find "Điều X" or "Điều X.Y", used for hierarchical splitting.
# re.split keeps the delimiter, which is what we want.
CLAUSE_PATTERN = re.compile(r"(Điều\s+\d+(?:\.\d+)*)")
CHUNK_SIZE_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 50

# ---------------------------------------------------------------------------
# 1. DOCUMENT LOADING
# ---------------------------------------------------------------------------
def load_document(path: str) -> str:
    """Loads a document as a single raw string."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise Exception(f"Document not found at: {path}")
    except Exception as e:
        raise Exception(f"Error loading document at {path}: {e}")

# ---------------------------------------------------------------------------
# 2.  CHUNKING STRATEGY (hierarchical, token‑aware)
# ---------------------------------------------------------------------------
def chunk_document(raw_text: str, source: str) -> List[Document]:
    """Split the raw text hierarchically and then token‑chunk it.

    1. Split on Vietnamese legal clause headings (Điều X, Điều 4.3 …)
    2. Further split each clause with a token‑aware splitter (TokenTextSplitter)
    """

    # First split by clause heading. The regex split with a capturing group
    # results in a list like: [text_before, delimiter, text_after, ...].
    # We then merge the delimiter with the text that follows it.
    raw_clauses = CLAUSE_PATTERN.split(raw_text)
    clauses = []
    for i in range(1, len(raw_clauses), 2):
        clause_text = raw_clauses[i] + raw_clauses[i+1]
        clauses.append(clause_text.strip())
    
    # Add the text before the first clause if it exists
    if raw_clauses[0].strip():
        clauses.insert(0, raw_clauses[0].strip())

    splitter = TokenTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        encoding_name="cl100k_base",  # same as OpenAI tiktoken encoding
    )

    chunked_docs: List[Document] = []
    for idx, clause in enumerate(clauses, start=1):
        # Extract heading for metadata (first line beginning with "Điều")
        heading_match = re.match(r"Điều\s+\d+(?:\.\d+)*", clause)
        heading = heading_match.group(0) if heading_match else f"Clause {idx}"

        # Token‑split the clause
        sub_chunks = splitter.split_text(clause)
        for sub_idx, chunk_text in enumerate(sub_chunks, start=1):
            metadata = {
                "heading": heading,
                "clause_index": idx,
                "sub_index": sub_idx,
                "source": source,
            }
            chunked_docs.append(Document(page_content=chunk_text, metadata=metadata))

    return chunked_docs

# ---------------------------------------------------------------------------
# 3.  EMBEDDING VIA DIRECT HTTP CALL (no LangChain wrapper)
# ---------------------------------------------------------------------------
def _embed_text_via_http(text: str) -> Union[List[float], None]:
    payload = {"input": text, "model": EMBEDDING_MODEL_NAME}
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(EMBEDDINGS_API_URL, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            data = r.json()
            if "data" in data and data["data"] and "embedding" in data["data"][0]:
                 return data["data"][0]["embedding"]
        print(f"⚠️  Embedding request failed ({r.status_code}): {r.text[:120]}")
    except Exception as e:
        print(f"⚠️  Exception in embedding request: {e}")
    return None

# ---------------------------------------------------------------------------
# 4.  QDRANT INGESTION
# ---------------------------------------------------------------------------
def ingest_chunks_into_qdrant(client: QdrantClient, collection: str, docs: List[Document]):
    if not docs:
        print("No document chunks to ingest.")
        return

    # Determine vector size with the first chunk
    first_vec = _embed_text_via_http(docs[0].page_content[:200])
    if not first_vec:
        print("❌  Cannot determine embedding dimension → abort.")
        return
    dimension = len(first_vec)

    # Ensure collection exists
    try:
        client.get_collection(collection_name=collection)
        print(f"✅  Collection '{collection}' already exists.")
    except Exception:
        print(f"ℹ️   Creating collection '{collection}' (dim={dimension}) …")
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )

    # Prepare points
    points: List[models.PointStruct] = []
    print(f"Embedding {len(docs)} chunks...")
    for doc in docs:
        vec = _embed_text_via_http(doc.page_content)
        if not vec:
            print(f"  - Skipping chunk with heading '{doc.metadata.get('heading')}' due to embedding failure.")
            continue
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={**doc.metadata, "text": doc.page_content},
            )
        )

    if not points:
        print("❌ No chunks were successfully embedded. Aborting upsert.")
        return

    # Upsert in batches
    BATCH = 100
    print(f"Upserting {len(points)} points to Qdrant...")
    for i in range(0, len(points), BATCH):
        batch = points[i : i + BATCH]
        client.upsert(collection_name=collection, points=batch, wait=True)
        print(f"  • Upserted batch {(i//BATCH)+1}/{((len(points)-1)//BATCH)+1} (size={len(batch)})")

# ---------------------------------------------------------------------------
# 5.  MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    print("\n——— Policy document ingestion started ———\n")

    # 5.1 Connect to Qdrant
    try:
        qdrant = QdrantClient(url=QDRANT_URL)
        qdrant.get_collections()
        print(f"✅  Connected to Qdrant @ {QDRANT_URL}")
    except Exception as e:
        print(f"❌  Cannot connect to Qdrant: {e}")
        return

    # 5.2 Load raw text
    doc_path = os.path.join(POLICY_DOCS_PATH, POLICY_FILENAME)
    try:
        raw_text = load_document(doc_path)
        print(f"✅  Loaded document '{POLICY_FILENAME}' (len={len(raw_text):,} chars)")
    except Exception as e:
        print(f"❌  {e}")
        return

    # 5.3 Chunk it (hierarchical & token‑aware)
    chunks = chunk_document(raw_text, source=POLICY_FILENAME)
    print(f"✅  Produced {len(chunks)} chunks (token size={CHUNK_SIZE_TOKENS}, overlap={CHUNK_OVERLAP_TOKENS})")

    # 5.4 Log chunks to chunked.txt for inspection
    chunks_file = os.path.join(POLICY_DOCS_PATH, "chunked.txt")
    try:
        with open(chunks_file, "w", encoding="utf-8") as f:
            for i, d in enumerate(chunks, start=1):
                f.write(
                    f"--- Chunk {i} | heading: {d.metadata['heading']} | size: {len(d.page_content)} chars ---\n"
                )
                f.write(d.page_content.strip())
                f.write("\n\n")
        print(f"✅  Wrote chunk log → {chunks_file}")
    except Exception as e:
        print(f"⚠️  Could not write chunk log: {e}")

    # 5.5 Embed & ingest
    ingest_chunks_into_qdrant(qdrant, QDRANT_COLLECTION_NAME, chunks)

    print("\n——— Ingestion finished ———")


if __name__ == "__main__":
    main()
