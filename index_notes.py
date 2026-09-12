from pathlib import Path

import pandas as pd

from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


# ======================================================
# CONFIGURATION
# ======================================================

NOTES_PATH = Path("data/context_notes.csv")

QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "context_notes"

EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"


# ======================================================
# LOAD AND PREPARE DOCUMENTS
# ======================================================

def load_documents():
    """
    Load context_notes.csv using LangChain CSVLoader
    and add metadata required for filtering.
    """

    if not NOTES_PATH.exists():
        raise FileNotFoundError(
            f"Context notes file not found: {NOTES_PATH}"
        )

    # Load CSV using LangChain
    loader = CSVLoader(
        file_path=str(NOTES_PATH)
    )

    documents = loader.load()

    # Read CSV with pandas so we can prepare metadata
    notes_df = pd.read_csv(NOTES_PATH)

    required_columns = {
    "note_id",
    "date",
    "applies_to",
    "note",
}

    missing = required_columns - set(notes_df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    notes_df["date"] = pd.to_datetime(
        notes_df["date"],
        errors="coerce",
    )

    if notes_df["date"].isna().any():
        raise ValueError(
            "Invalid date found in context_notes.csv"
        )

    if len(documents) != len(notes_df):
        raise ValueError(
            "Number of loaded documents does not match "
            "number of CSV rows."
        )

    # Add metadata to each document
    for document, (_, row) in zip(
        documents,
        notes_df.iterrows(),
    ):
        document.metadata.update(
            {
                "note_id": str(row["note_id"]),
                "route": str(row["applies_to"]).strip(),
                "date": row["date"].date().isoformat(),
                "date_timestamp": int(
                    row["date"].timestamp()
                ),
            }
        )

    return documents


# ======================================================
# CREATE EMBEDDINGS
# ======================================================

def create_embeddings():
    """
    Create the Nomic embedding model.
    """

    print("Loading Nomic embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "trust_remote_code": True
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    return embeddings


# ======================================================
# INDEX DOCUMENTS INTO QDRANT
# ======================================================

def index_documents(documents, embeddings):
   
    """
    Convert documents into vectors and store them
    in the Qdrant Docker server.
    """

    print("Connecting to Qdrant...")

    client = QdrantClient(url=QDRANT_URL)

    # Delete old collection so we start with clean metadata
    if client.collection_exists(COLLECTION_NAME):
        print(f"Deleting existing collection: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)

    print("Indexing documents into Qdrant...")

    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )

    print("Indexing completed.")

    # --------------------------------------------------
    # DEBUG: Inspect what was actually stored
    # --------------------------------------------------

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=20,
        with_payload=True,
        with_vectors=False,
    )

    print("\nStored Qdrant payloads:")

    for point in points:
        print(point.payload)

    return vector_store


# ======================================================
# MAIN
# ======================================================

def main():

    print("=" * 60)
    print("FreightTiger Context Notes Indexing")
    print("=" * 60)

    # 1. Load documents
    print("\n[1/3] Loading context notes...")
    documents = load_documents()

    print(
        f"Loaded {len(documents)} context notes."
    )

    # 2. Create embeddings
    print("\n[2/3] Creating Nomic embeddings...")
    embeddings = create_embeddings()

    # 3. Store in Qdrant
    print("\n[3/3] Indexing into Qdrant...")
    index_documents(
        documents,
        embeddings,
    )

    print("\n" + "=" * 60)
    print("Indexing completed successfully!")
    print("=" * 60)

    print(
        f"\nCollection: {COLLECTION_NAME}"
    )

    print(
        f"Qdrant: {QDRANT_URL}"
    )

    print(
        f"Documents indexed: {len(documents)}"
    )


if __name__ == "__main__":
    main()