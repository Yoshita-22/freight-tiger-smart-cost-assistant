from pathlib import Path
import sys

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# ---------------------------------------------------------
# Make src/ importable when running:
# uv run python tests/eval_evidence.py
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from freight_tiger_smart_cost_assistant.retrieval import ContextRetriever
from freight_tiger_smart_cost_assistant.evidence_validation import (
    validate_evidence,
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

GOLDEN_PATH = PROJECT_ROOT / "data" / "eval_evidence_golden.csv"
NOTES_PATH = PROJECT_ROOT / "data" / "context_notes.csv"


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Create LLM
# ---------------------------------------------------------

def create_llm():

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
    )

    return [
        ("Gemini", llm)
    ]


# ---------------------------------------------------------
# Load golden dataset
# ---------------------------------------------------------

def load_golden_dataset():

    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"Golden dataset not found: {GOLDEN_PATH}"
        )

    golden = pd.read_csv(GOLDEN_PATH)

    required_columns = {
        "case_id",
        "route",
        "week_of",
        "expected_support",
        "expected_note_id",
    }

    missing = required_columns - set(golden.columns)

    if missing:
        raise ValueError(
            f"Golden dataset missing columns: {sorted(missing)}"
        )

    return golden


# ---------------------------------------------------------
# Evaluate one case
# ---------------------------------------------------------

def evaluate_case(
    llm,
    retriever,
    row,
):

    route = str(row["route"]).strip()
    week_of = str(row["week_of"]).strip()

    expected_support = bool(row["expected_support"])

    expected_note_id = row["expected_note_id"]

    if pd.isna(expected_note_id):
        expected_note_id = ""

    expected_note_id = str(expected_note_id).strip()

    # -----------------------------------------------------
    # Retrieve relevant context
    # -----------------------------------------------------

    retrieved_notes = retriever.retrieve(
        route=route,
        week_of=week_of,
        top_k=5,
    )

    # -----------------------------------------------------
    # Run actual LLM evidence validation
    #
    # We don't have real cost metrics here because this
    # evaluation focuses on evidence correctness.
    # -----------------------------------------------------

    result = validate_evidence(
        llms=llm,
        route=route,
        week_of=week_of,
        cost_per_tonne_km=0.0,
        vs_own_history=0.0,
        vs_similar_routes=0.0,
        retrieved_notes=retrieved_notes,
    )

    actual_support = bool(
        result.get("supports_cost_increase", False)
    )

    actual_note_id = str(
        result.get("matched_note_id", "")
    ).strip()

    # -----------------------------------------------------
    # Compare with golden labels
    # -----------------------------------------------------

    support_correct = (
        actual_support == expected_support
    )

    if expected_support:
        citation_correct = (
            actual_note_id == expected_note_id
        )
    else:
        # For an unexplained case, there should be
        # no supporting note.
        citation_correct = (
            actual_note_id == ""
        )

    case_correct = (
        support_correct and citation_correct
    )

    return {
        "case_id": row["case_id"],
        "route": route,
        "week_of": week_of,
        "expected_support": expected_support,
        "actual_support": actual_support,
        "expected_note_id": expected_note_id,
        "actual_note_id": actual_note_id,
        "support_correct": support_correct,
        "citation_correct": citation_correct,
        "case_correct": case_correct,
        "reason": result.get("reason", ""),
    }


# ---------------------------------------------------------
# Calculate evaluation metrics
# ---------------------------------------------------------

def calculate_metrics(results):

    total = len(results)

    support_correct = sum(
        r["support_correct"]
        for r in results
    )

    citation_correct = sum(
        r["citation_correct"]
        for r in results
    )

    case_correct = sum(
        r["case_correct"]
        for r in results
    )

    # -----------------------------------------------------
    # False justification:
    #
    # Golden says there is NO valid explanation,
    # but the system says there IS one.
    # -----------------------------------------------------

    false_justifications = sum(
        1
        for r in results
        if (
            r["expected_support"] is False
            and r["actual_support"] is True
        )
    )

    # -----------------------------------------------------
    # False negative:
    #
    # Golden says evidence exists,
    # but system fails to find it.
    # -----------------------------------------------------

    false_negatives = sum(
        1
        for r in results
        if (
            r["expected_support"] is True
            and r["actual_support"] is False
        )
    )

    return {
        "total_cases": total,
        "support_decision_accuracy": (
            support_correct / total
            if total
            else 0
        ),
        "citation_accuracy": (
            citation_correct / total
            if total
            else 0
        ),
        "overall_case_accuracy": (
            case_correct / total
            if total
            else 0
        ),
        "false_justification_rate": (
            false_justifications / total
            if total
            else 0
        ),
        "false_negative_rate": (
            false_negatives / total
            if total
            else 0
        ),
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("EVIDENCE GROUNDING GOLDEN-SET EVALUATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Load golden dataset
    # -----------------------------------------------------

    golden = load_golden_dataset()

    print(f"\nGolden cases: {len(golden)}")

    # -----------------------------------------------------
    # Create retriever
    # -----------------------------------------------------

    print("\nInitializing retriever...")

    retriever = ContextRetriever(
        notes_path=NOTES_PATH
    )

    # -----------------------------------------------------
    # Create LLM
    # -----------------------------------------------------

    print("\nInitializing LLM...")

    llm = create_llm()

    # -----------------------------------------------------
    # Run evaluation
    # -----------------------------------------------------

    results = []

    for _, row in golden.iterrows():

        print("\n" + "-" * 70)

        print(
            f"Case {row['case_id']}: "
            f"{row['route']} | {row['week_of']}"
        )

        try:

            result = evaluate_case(
                llm=llm,
                retriever=retriever,
                row=row,
            )

            results.append(result)

            if result["case_correct"]:
                status = "PASS"
            else:
                status = "FAIL"

            print(f"Expected support : {result['expected_support']}")
            print(f"Actual support   : {result['actual_support']}")
            print(f"Expected note    : {result['expected_note_id'] or 'None'}")
            print(f"Actual note      : {result['actual_note_id'] or 'None'}")
            print(f"Result           : {status}")

            print(
                f"Reason           : "
                f"{result['reason']}"
            )

        except Exception as exc:

            print("\nERROR running case:")
            print(type(exc).__name__)
            print(exc)

            results.append({
                "case_id": row["case_id"],
                "route": row["route"],
                "week_of": row["week_of"],
                "expected_support": bool(
                    row["expected_support"]
                ),
                "actual_support": False,
                "expected_note_id": (
                    ""
                    if pd.isna(row["expected_note_id"])
                    else str(row["expected_note_id"])
                ),
                "actual_note_id": "",
                "support_correct": False,
                "citation_correct": False,
                "case_correct": False,
                "reason": "Evaluation error",
            })

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    metrics = calculate_metrics(results)

    print("\n")
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"\nTotal cases: "
        f"{metrics['total_cases']}"
    )

    print(
        f"Support decision accuracy: "
        f"{metrics['support_decision_accuracy']:.2%}"
    )

    print(
        f"Citation accuracy: "
        f"{metrics['citation_accuracy']:.2%}"
    )

    print(
        f"Overall case accuracy: "
        f"{metrics['overall_case_accuracy']:.2%}"
    )

    print(
        f"False justification rate: "
        f"{metrics['false_justification_rate']:.2%}"
    )

    print(
        f"False negative rate: "
        f"{metrics['false_negative_rate']:.2%}"
    )

    # -----------------------------------------------------
    # Detailed results table
    # -----------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\n")
    print("=" * 70)
    print("CASE-BY-CASE RESULTS")
    print("=" * 70)

    print(
        results_df[
            [
                "case_id",
                "expected_support",
                "actual_support",
                "expected_note_id",
                "actual_note_id",
                "case_correct",
            ]
        ].to_string(index=False)
    )

    # -----------------------------------------------------
    # Save evaluation results
    # -----------------------------------------------------

    output_path = PROJECT_ROOT / "logs" / "evidence_eval_results.csv"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nDetailed results saved to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()