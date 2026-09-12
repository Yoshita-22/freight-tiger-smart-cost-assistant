from pathlib import Path

import pandas as pd

from .preprocessing import preprocess_shipments
from .weekly_metrics import calculate_weekly_metrics
from .anomaly_Detection import detect_anomalies
from .retrieval import ContextRetriever
from .evidence_validation import validate_evidence
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()




# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_SHIPMENT_PATH = Path(
    "data/shipment_records.csv"
)

DEFAULT_NOTES_PATH = Path(
    "data/context_notes.csv"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/output.csv"
)

DEFAULT_QDRANT_PATH = Path(
    "data/qdrant"
)

ANOMALY_THRESHOLD_PCT = 10.0


# ============================================================
# LLM CREATION
# ============================================================

def create_llms():
    """
    Create the LLM used for evidence validation.

    Keep the actual provider configuration here so the rest
    of the pipeline does not depend on a particular LLM.

    Replace this implementation with the provider/model
    you choose for the assignment.
    """

    


    gemini = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    groq = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    return [
        ("Gemini", gemini),
        ("Groq", groq),
    ]


# ============================================================
# FINAL REASON FOR NORMAL ROUTES
# ============================================================

def build_normal_reason(
    vs_own_history: float,
    vs_similar_routes: float,
) -> str:
    """
    Generate a deterministic explanation for a route-week
    that was not anomalous.

    No LLM or RAG is required here.
    """

    if pd.isna(vs_own_history) and pd.isna(
        vs_similar_routes
    ):
        return (
            "No historical or similar-route baseline "
            "was available for comparison."
        )

    comparisons = []

    if not pd.isna(vs_own_history):
        comparisons.append(
            f"{vs_own_history:.1f}% versus its own "
            "historical baseline"
        )

    if not pd.isna(vs_similar_routes):
        comparisons.append(
            f"{vs_similar_routes:.1f}% versus similar routes"
        )

    comparison_text = " and ".join(comparisons)

    return (
        f"Cost is within the expected range based on "
        f"{comparison_text}."
    )


# ============================================================
# FINAL REASON FOR UNEXPLAINED ANOMALY
# ============================================================

def build_unexplained_reason(
    vs_own_history: float,
    vs_similar_routes: float,
    evidence_reason: str,
) -> str:
    """
    Build the final reason when an anomaly has no valid
    supporting explanation.
    """

    comparisons = []

    if not pd.isna(vs_own_history):
        comparisons.append(
            f"{vs_own_history:.1f}% versus its own "
            "historical baseline"
        )

    if not pd.isna(vs_similar_routes):
        comparisons.append(
            f"{vs_similar_routes:.1f}% versus similar routes"
        )

    if comparisons:
        comparison_text = " and ".join(comparisons)

        base_reason = (
            f"Cost is unusually high based on "
            f"{comparison_text}."
        )

    else:
        base_reason = (
            "Cost was identified as anomalous based on "
            "the available comparison baseline."
        )

    if evidence_reason:
        return f"{base_reason} {evidence_reason}"

    return (
        f"{base_reason} No applicable context note "
        "was found to explain the increase."
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    shipment_path: str | Path = DEFAULT_SHIPMENT_PATH,
    notes_path: str | Path = DEFAULT_NOTES_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    qdrant_path: str | Path = DEFAULT_QDRANT_PATH,
    anomaly_threshold_pct: float = ANOMALY_THRESHOLD_PCT,
):
    """
    Run the complete FreightTiger smart cost assistant.

    Steps:

        1. Load and preprocess shipment data.
        2. Calculate weekly route-level metrics.
        3. Calculate anomaly comparisons.
        4. For anomalous route-weeks only:
              - retrieve context using RAG
              - validate evidence using LLM
        5. Produce final flagged decision.
        6. Save exact required CSV format.
    """

    shipment_path = Path(shipment_path)
    notes_path = Path(notes_path)
    output_path = Path(output_path)
    qdrant_path = Path(qdrant_path)

    # ========================================================
    # STEP 1 — PREPROCESS SHIPMENT DATA
    # ========================================================

    print("Loading and preprocessing shipment data...")

    shipments = preprocess_shipments(
        shipment_path
    )

    print(
        f"Loaded {len(shipments)} valid shipment records."
    )

    # ========================================================
    # STEP 2 — WEEKLY METRICS
    # ========================================================

    print("Calculating weekly route metrics...")

    weekly = calculate_weekly_metrics(
        shipments
    )

    print(
        f"Created {len(weekly)} route-week observations."
    )

    # ========================================================
    # STEP 3 — ANOMALY DETECTION
    # ========================================================

    print("Calculating anomaly baselines...")

    weekly = detect_anomalies(
        weekly,
        threshold_pct=anomaly_threshold_pct,
    )

    # IMPORTANT:
    #
    # detect_anomalies() should create:
    #
    #     is_anomalous
    #

    if "is_anomalous" not in weekly.columns:

        raise ValueError(
            "anomaly_detection.py must create "
            "`is_anomalous` before the RAG stage."
        )

    anomaly_count = int(
        weekly["is_anomalous"].sum()
    )

    print(
        f"Found {anomaly_count} anomalous route-week "
        "candidates."
    )

    # ========================================================
    # STEP 4 — INITIALIZE RAG
    # ========================================================

    retriever = None
    llm = None

    # IMPORTANT:
    #
    # If there are no anomalies, there is no reason to
    # initialize/call RAG or the LLM.
    #
    if anomaly_count > 0:

        print("Initializing RAG components...")

        retriever = ContextRetriever(
            notes_path=notes_path
            
        )

        llms = create_llms()

    # ========================================================
    # STEP 5 — PROCESS EVERY ROUTE-WEEK
    # ========================================================

    results = []

    for _, row in weekly.iterrows():

        route = (
            f"{row['origin']}-{row['destination']}"
        )

        week_of = pd.Timestamp(
            row["week_of"]
        )

        current_cost = float(
            row["cost_per_tonne_km"]
        )

        vs_own_history = row[
            "vs_own_history"
        ]

        vs_similar_routes = row[
            "vs_similar_routes"
        ]

        is_anomalous = bool(
            row["is_anomalous"]
        )

        # ----------------------------------------------------
        # CASE 1 — NOT ANOMALOUS
        # ----------------------------------------------------

        if not is_anomalous:

            flagged = "No"

            matched_note_id = ""

            reason = build_normal_reason(
                vs_own_history=
                    vs_own_history,
                vs_similar_routes=
                    vs_similar_routes,
            )

        # ----------------------------------------------------
        # CASE 2 — ANOMALOUS
        # ----------------------------------------------------

        else:

            print(
                f"Analyzing anomaly: "
                f"{route} | {week_of.date()}"
            )

            # ----------------------------------------------
            # RAG RETRIEVAL
            # ----------------------------------------------

            retrieved_notes = retriever.retrieve(
                route=route,
                week_of=week_of,
                top_k=3,
            )
            
            # ----------------------------------------------
            # NO RETRIEVED DOCUMENT
            # ----------------------------------------------
            print("\nRETRIEVED NOTES")
            print("Route:", route)
            print("Week:", week_of.date())

            for note in retrieved_notes:
                print(
                    note["note_id"],
                    "|",
                    note["date"],
                    "|",
                    note["route"],
                    "|",
                    note["note"],
                    "| score:",
                    note["similarity_score"],
                )

            if not retrieved_notes:

                flagged = "Yes"

                matched_note_id = ""

                reason = build_unexplained_reason(
                    vs_own_history=
                        vs_own_history,
                    vs_similar_routes=
                        vs_similar_routes,
                    evidence_reason=(
                        "No applicable context note "
                        "was found to explain the "
                        "increase."
                    ),
                )

            # ----------------------------------------------
            # DOCUMENT(S) RETRIEVED
            # ----------------------------------------------

            else:

                evidence = validate_evidence(
                    llms=llms,
                    route=route,
                    week_of=str(
                        week_of.date()
                    ),
                    cost_per_tonne_km=
                        current_cost,
                    vs_own_history=
                        vs_own_history,
                    vs_similar_routes=
                        vs_similar_routes,
                    retrieved_notes=
                        retrieved_notes,
                )

                # ------------------------------------------
                # VALID EXPLANATION FOUND
                # ------------------------------------------

                if evidence[
                    "supports_cost_increase"
                ]:

                    flagged = "No"

                    matched_note_id = evidence[
                        "matched_note_id"
                    ]

                    reason = evidence[
                        "reason"
                    ]

                # ------------------------------------------
                # DOCUMENT FOUND BUT NOT A VALID
                # EXPLANATION
                # ------------------------------------------

                else:

                    flagged = "Yes"

                    matched_note_id = ""

                    reason = build_unexplained_reason(
                        vs_own_history=
                            vs_own_history,
                        vs_similar_routes=
                            vs_similar_routes,
                        evidence_reason=
                            evidence["reason"],
                    )

        # ----------------------------------------------------
        # STORE FINAL RESULT
        # ----------------------------------------------------

        results.append(
            {
                "route": route,
                "week_of": week_of.date().isoformat(),
                "cost_per_tonne_km": current_cost,
                "vs_own_history": (
                    None
                    if pd.isna(vs_own_history)
                    else round(
                        float(vs_own_history),
                        2,
                    )
                ),
                "vs_similar_routes": (
                    None
                    if pd.isna(vs_similar_routes)
                    else round(
                        float(vs_similar_routes),
                        2,
                    )
                ),
                "flagged": flagged,
                "matched_note_id": matched_note_id,
                "reason": reason,
            }
        )

    # ========================================================
    # STEP 6 — CREATE FINAL OUTPUT
    # ========================================================

    output = pd.DataFrame(results)

    required_columns = [
        "route",
        "week_of",
        "cost_per_tonne_km",
        "vs_own_history",
        "vs_similar_routes",
        "flagged",
        "matched_note_id",
        "reason",
    ]

    output = output[
        required_columns
    ]

    # ========================================================
    # STEP 7 — SAVE CSV
    # ========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        output_path,
        index=False,
    )

    print()
    print("Pipeline completed.")
    print(
        f"Output written to: {output_path}"
    )
    print(
        f"Output rows: {len(output)}"
    )

    return output