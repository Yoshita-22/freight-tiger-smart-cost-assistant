from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# ============================================================
# EVIDENCE PROMPT
# ============================================================

def build_evidence_prompt() -> ChatPromptTemplate:
    """
    Prompt used to determine whether retrieved context
    genuinely explains an anomalous freight-cost increase.
    """

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an evidence-validation assistant for a freight
cost monitoring system.

Your job is NOT to determine whether the route is anomalous.
That has already been determined using deterministic
calculations.

Your job is ONLY to determine whether the retrieved context
notes provide a genuine, documented explanation for the
higher freight cost.

STRICT RULES:

1. The note must apply to the requested route, or explicitly
   apply to all routes/nationwide.

2. The note must be temporally applicable to the requested week.

3. The note's publication/date field does NOT automatically
   define the only week to which the note applies.

4. Carefully interpret temporal expressions in the note such as:
   - "from Feb 24 to Mar 8"
   - "until March 8"
   - "through March 8"
   - "this week"
   - "this month"
   - "this quarter"
   - "starting this week"

5. If the note explicitly or clearly states that an event affected
   the requested week, it may be considered temporally applicable.
   A note is valid evidence only if its stated time period explicitly 
   overlaps the anomalous week, or if the note clearly states that the condition
   remains in effect beyond its publication date. Do not assume that an event continues 
   indefinitely.

6. The note must provide evidence of a factor that could increase
   freight, transportation, or shipping cost.

7. Do NOT infer causality from a note that merely describes an event.

8. A note saying that conditions were normal, demand was stable,
   or there were no significant disruptions does NOT explain a
   price increase.

9. If none of the notes clearly explains the cost increase,
   return supports_cost_increase=false.

10. If supports_cost_increase=true, matched_note_id MUST be
    one of the retrieved note IDs.

11. Do not invent facts, dates, events, routes, or note IDs.

12. Base the reason ONLY on the retrieved context notes.

Return ONLY valid JSON with this exact structure:

{{
    "supports_cost_increase": true or false,
    "matched_note_id": "note ID or empty string",
    "reason": "short explanation based only on the retrieved note"
}}
""",
            ),
            (
                "human",
                """
Route:
{route}

Week:
{week_of}

Current cost per tonne-km:
{cost_per_tonne_km}

Increase versus own 8-week history:
{vs_own_history}%

Increase versus similar routes:
{vs_similar_routes}%

Retrieved context notes:

{retrieved_notes}
""",
            ),
        ]
    )


# ============================================================
# FORMAT RETRIEVED NOTES
# ============================================================

def format_retrieved_notes(
    retrieved_notes: list[dict[str, Any]]
) -> str:
    """
    Convert retrieved notes into text for the LLM.
    """

    if not retrieved_notes:
        return "No context notes were retrieved."

    formatted_notes = []

    for note in retrieved_notes:

        formatted_notes.append(
            f"""
Note ID: {note["note_id"]}
Date: {note["date"]}
Route: {note["route"]}
Content: {note["note"]}
Similarity score: {note.get("similarity_score", "N/A")}
""".strip()
        )

    return "\n\n---\n\n".join(formatted_notes)


# ============================================================
# NORMALIZE BOOLEAN
# ============================================================

def normalize_boolean(value: Any) -> bool:
    """
    Safely convert the LLM's supports_cost_increase value
    into a Python boolean.

    Prevents:
        bool("false") == True
    """

    if isinstance(value, bool):
        return value

    if isinstance(value, str):

        value = value.strip().lower()

        if value in {"true", "yes", "1"}:
            return True

        if value in {"false", "no", "0", ""}:
            return False

    return False


# ============================================================
# VALIDATE LLM RESPONSE
# ============================================================

def validate_evidence_response(
    llm_response: dict[str, Any],
    retrieved_notes: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Validate and sanitize the LLM's evidence decision.

    The LLM is NOT trusted blindly.

    A note is accepted only if:

        1. The LLM says it supports the cost increase.
        2. The returned note ID actually exists in the
           retrieved documents.
    """

    if not isinstance(llm_response, dict):

        return {
            "supports_cost_increase": False,
            "matched_note_id": "",
            "reason": (
                "Invalid evidence-validation response. "
                "No supporting note was accepted."
            ),
        }

    supports = normalize_boolean(
        llm_response.get(
            "supports_cost_increase",
            False,
        )
    )

    matched_note_id = str(
        llm_response.get(
            "matched_note_id",
            "",
        )
    ).strip()

    reason = str(
        llm_response.get(
            "reason",
            "",
        )
    ).strip()

    retrieved_note_ids = {
        str(note["note_id"]).strip()
        for note in retrieved_notes
    }

    # --------------------------------------------------------
    # CASE 1 — No supporting evidence
    # --------------------------------------------------------

    if not supports:

        return {
            "supports_cost_increase": False,
            "matched_note_id": "",
            "reason": (
                reason
                or
                "No retrieved context note provides a "
                "clear explanation for the cost increase."
            ),
        }

    # --------------------------------------------------------
    # CASE 2 — LLM invented a note ID
    # --------------------------------------------------------

    if matched_note_id not in retrieved_note_ids:

        return {
            "supports_cost_increase": False,
            "matched_note_id": "",
            "reason": (
                "The model identified supporting evidence "
                "that was not present in the retrieved notes. "
                "No supporting note was accepted."
            ),
        }

    # --------------------------------------------------------
    # CASE 3 — Valid supporting evidence
    # --------------------------------------------------------

    return {
        "supports_cost_increase": True,
        "matched_note_id": matched_note_id,
        "reason": (
            reason
            or
            "The retrieved context provides a documented "
            "explanation for the cost increase."
        ),
    }


# ============================================================
# EVIDENCE VALIDATION
# ============================================================

def validate_evidence(
    llms: list[tuple[str, Any]],
    route: str,
    week_of: str,
    cost_per_tonne_km: float,
    vs_own_history: float,
    vs_similar_routes: float,
    retrieved_notes: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Validate whether retrieved context genuinely explains
    an anomalous freight-cost increase.

    LLMs are tried in order.

    Example:

        [
            ("Gemini", gemini),
            ("Groq", groq),
        ]

    If the first LLM fails because of a rate limit or
    another API error, the next LLM is attempted.

    If all LLMs fail, the system fails closed.
    """

    # ========================================================
    # NO RETRIEVED EVIDENCE
    # ========================================================

    if not retrieved_notes:

        return {
            "supports_cost_increase": False,
            "matched_note_id": "",
            "reason": (
                "No applicable context note was found "
                "to explain the cost increase."
            ),
            "llm_model": "none",
        }

    # ========================================================
    # BUILD PROMPT
    # ========================================================

    prompt = build_evidence_prompt()

    parser = JsonOutputParser()

    retrieved_text = format_retrieved_notes(
        retrieved_notes
    )

    # ========================================================
    # TRY LLMs IN ORDER
    # ========================================================

    for model_name, llm in llms:

        print(
            f"\nTrying evidence validation with: "
            f"{model_name}"
        )

        chain = prompt | llm | parser

        try:

            response = chain.invoke(
                {
                    "route": route,
                    "week_of": week_of,
                    "cost_per_tonne_km": (
                        f"{cost_per_tonne_km:.4f}"
                    ),
                    "vs_own_history": (
                        f"{vs_own_history:.2f}"
                    ),
                    "vs_similar_routes": (
                        f"{vs_similar_routes:.2f}"
                    ),
                    "retrieved_notes": retrieved_text,
                }
            )

            print(
                f"Evidence validation succeeded "
                f"using {model_name}"
            )

            # ------------------------------------------------
            # Validate response deterministically
            # ------------------------------------------------

            result = validate_evidence_response(
                llm_response=response,
                retrieved_notes=retrieved_notes,
            )

            result["llm_model"] = model_name

            return result

        except Exception as exc:

            # ------------------------------------------------
            # IMPORTANT:
            # Do NOT convert this immediately into
            # "unexplained".
            #
            # Try the next provider first.
            # ------------------------------------------------

            print(
                f"{model_name} failed:"
            )

            print(
                f"  Error type: {type(exc).__name__}"
            )

            print(
                f"  Error: {exc}"
            )

            print(
                "Trying next LLM..."
            )

    # ========================================================
    # ALL LLMs FAILED
    # ========================================================

    return {
        "supports_cost_increase": False,
        "matched_note_id": "",
        "reason": (
            "Evidence validation failed with all "
            "configured LLMs; no supporting note was accepted."
        ),
        "llm_model": "none",
    }