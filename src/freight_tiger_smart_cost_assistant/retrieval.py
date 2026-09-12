from pathlib import Path
from typing import Any

import pandas as pd

from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)


# ======================================================
# CONFIGURATION
# ======================================================

COLLECTION_NAME = "context_notes"

EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"

# Qdrant Docker server
QDRANT_URL = "http://localhost:6333"


# ======================================================
# CONTEXT RETRIEVER
# ======================================================

class ContextRetriever:
    """
    Retrieves relevant freight context notes using:

        1. Route/scope metadata filtering
        2. Semantic similarity using Nomic embeddings

    Temporal applicability is NOT used as a hard Qdrant
    filter. This is intentional.

    The note's actual content may contain temporal expressions
    such as:

        - "from Feb 24 to Mar 8"
        - "until March 8"
        - "this week"
        - "this quarter"

    These are evaluated later by the evidence-validation
    step.
    """

    def __init__(
        self,
        notes_path: str | Path,
    ):
        self.notes_path = Path(notes_path)

        if not self.notes_path.exists():
            raise FileNotFoundError(
                f"Context notes file not found: "
                f"{self.notes_path}"
            )

        # --------------------------------------------------
        # 1. Load CSV using LangChain CSVLoader
        # --------------------------------------------------

        loader = CSVLoader(
            file_path=str(self.notes_path)
        )

        self.documents = loader.load()

        # --------------------------------------------------
        # 2. Add metadata
        # --------------------------------------------------

        self._add_metadata()

        # --------------------------------------------------
        # 3. Create Nomic embedding model
        # --------------------------------------------------

        print("Loading Nomic embedding model...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                "trust_remote_code": True
            },
            encode_kwargs={
                "normalize_embeddings": True
            },
        )

        # --------------------------------------------------
        # 4. Connect to Qdrant Docker server
        # --------------------------------------------------

        print(
            f"Connecting to Qdrant at {QDRANT_URL}..."
        )

        self.client = QdrantClient(
            url=QDRANT_URL
        )

        # --------------------------------------------------
        # 5. Connect to existing collection
        # --------------------------------------------------

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding=self.embeddings,
        )

    # ======================================================
    # METADATA
    # ======================================================

    def _add_metadata(self) -> None:
        """
        Add structured metadata to every context note.

        The CSV contains:

            note_id
            date
            applies_to
            note

        Qdrant metadata stores:

            note_id
            route
            date
            date_timestamp

        The CSV field "applies_to" is mapped to the metadata
        field "route".
        """

        notes_df = pd.read_csv(
            self.notes_path
        )

        required_columns = {
            "note_id",
            "date",
            "applies_to",
            "note",
        }

        missing = (
            required_columns
            - set(notes_df.columns)
        )

        if missing:
            raise ValueError(
                "Missing required columns in "
                "context_notes.csv: "
                f"{sorted(missing)}"
            )

        # --------------------------------------------------
        # Parse dates
        # --------------------------------------------------

        notes_df["date"] = pd.to_datetime(
            notes_df["date"],
            errors="coerce",
        )

        if notes_df["date"].isna().any():
            raise ValueError(
                "Invalid date found in "
                "context_notes.csv."
            )

        # --------------------------------------------------
        # Make sure CSV rows match loaded documents
        # --------------------------------------------------

        if len(notes_df) != len(self.documents):
            raise ValueError(
                "Number of CSV rows does not match "
                "number of loaded documents."
            )

        # --------------------------------------------------
        # Attach metadata
        # --------------------------------------------------

        for document, (_, row) in zip(
            self.documents,
            notes_df.iterrows(),
        ):

            document.metadata.update(
                {
                    "note_id": str(
                        row["note_id"]
                    ).strip(),

                    "route": str(
                        row["applies_to"]
                    ).strip(),

                    "date": row[
                        "date"
                    ].date().isoformat(),

                    "date_timestamp": int(
                        row["date"].timestamp()
                    ),
                }
            )

    # ======================================================
    # SEMANTIC SEARCH QUERY
    # ======================================================

    @staticmethod
    def build_query(
        route: str,
        week_of: str | pd.Timestamp,
    ) -> str:
        """
        Build a semantic-search query for an anomalous
        route-week.

        The week is included in the query so the embedding
        search can use temporal information present in the
        note text.

        The week is intentionally NOT used as a hard
        metadata filter.
        """

        week_of = pd.Timestamp(
            week_of
        )

        return (
            f"Shipping freight cost increase on the "
            f"{route} route during the week beginning "
            f"{week_of.date()}. "

            f"Find documented factors that could explain "
            f"an increase in freight, transportation, or "
            f"shipping cost, including fuel price changes, "
            f"diesel prices, tolls, weather, flooding, "
            f"festivals, surcharges, high demand, limited "
            f"truck availability, delays, disruptions, "
            f"road conditions, detours, or other logistics "
            f"cost drivers. "

            f"Pay attention to temporal expressions such as "
            f"'from', 'until', 'through', 'starting this week', "
            f"'this month', 'this quarter', or other duration "
            f"expressions."
        )

    # ======================================================
    # ROUTE FILTER
    # ======================================================

    @staticmethod
    def build_filter(
        route: str,
    ) -> Filter:
        """
        Restrict retrieval to:

            1. The requested route
            2. All Routes
            3. Nationwide

        No date filter is applied here.

        This allows notes such as:

            N001:
            dated Feb 24 but valid through Mar 8

        to be retrieved for an anomaly on Mar 3.
        """

        return Filter(
            should=[
                FieldCondition(
                    key="metadata.route",
                    match=MatchValue(
                        value=route
                    ),
                ),

                FieldCondition(
                    key="metadata.route",
                    match=MatchValue(
                        value="All Routes"
                    ),
                ),

                FieldCondition(
                    key="metadata.route",
                    match=MatchValue(
                        value="Nationwide"
                    ),
                ),
            ]
        )

    # ======================================================
    # RETRIEVE
    # ======================================================

    def retrieve(
        self,
        route: str,
        week_of: str | pd.Timestamp,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve potentially relevant context notes.

        Process:

            1. Build anomaly-specific semantic query.
            2. Filter by route/scope.
            3. Perform Nomic semantic similarity search.
            4. Return the top-k notes.

        Temporal applicability and whether the note actually
        explains the cost increase are evaluated later by
        evidence validation.
        """

        # --------------------------------------------------
        # Build semantic query
        # --------------------------------------------------

        query = self.build_query(
            route=route,
            week_of=week_of,
        )

        # --------------------------------------------------
        # Build route filter
        # --------------------------------------------------

        metadata_filter = self.build_filter(
            route=route,
        )

        # --------------------------------------------------
        # Debug information
        # --------------------------------------------------

        print(
            f"\nRetrieving context notes for:"
        )

        print(
            f"Route: {route}"
        )

        print(
            f"Week: {pd.Timestamp(week_of).date()}"
        )

        # --------------------------------------------------
        # Semantic search
        # --------------------------------------------------

        results = (
            self.vector_store
            .similarity_search_with_score(
                query=query,
                k=top_k,
                filter=metadata_filter,
            )
        )

        # --------------------------------------------------
        # Convert results to dictionaries
        # --------------------------------------------------

        retrieved_notes = []

        for document, score in results:

            metadata = document.metadata

            retrieved_notes.append(
                {
                    "note_id": metadata.get(
                        "note_id",
                        "",
                    ),

                    "date": metadata.get(
                        "date",
                        "",
                    ),

                    "route": metadata.get(
                        "route",
                        "",
                    ),

                    "note": document.page_content,

                    "similarity_score": float(
                        score
                    ),
                }
            )

        # --------------------------------------------------
        # Debug retrieved notes
        # --------------------------------------------------

        print(
            "\nRETRIEVED NOTES"
        )

        if not retrieved_notes:

            print(
                "No context notes retrieved."
            )

        else:

            for note in retrieved_notes:

                print(
                    f"\n"
                    f"Note ID: {note['note_id']}\n"
                    f"Route: {note['route']}\n"
                    f"Date: {note['date']}\n"
                    f"Similarity: "
                    f"{note['similarity_score']:.4f}\n"
                    f"Content: {note['note']}"
                )

        return retrieved_notes