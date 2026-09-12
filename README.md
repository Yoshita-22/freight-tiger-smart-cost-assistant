# FreightTiger Smart Cost Assistant

## Overview

This project implements a Smart Assistant that monitors shipping costs and identifies
route-week combinations where freight costs appear unusually high.

The system combines:

- Deterministic data processing and anomaly detection
- Weekly route-level cost analysis
- Own-route historical baselines
- Similar-route comparisons
- Retrieval-Augmented Generation (RAG)
- Qdrant vector search
- Local Nomic embeddings
- LLM-based evidence validation
- Deterministic guardrails against unsupported citations
- Automated evaluation using a manually labeled golden set
- Reproducibility and cost tracking

The key design principle is:

> Use deterministic logic for numerical decisions and use the LLM only where
> contextual reasoning is required.

---

# 1. Problem Statement

Freight costs can increase because of many factors such as:

- Fuel price changes
- Festivals and temporary surcharges
- Toll changes
- Flooding and weather disruptions
- Road conditions
- Detours
- Truck availability
- Logistics disruptions

The objective is to identify route-week combinations where the shipping cost is
unusually high and determine whether there is a documented contextual reason for
the increase.

The system should distinguish between:

1. A cost increase that has a documented explanation
2. A cost increase that does not have a documented explanation

An explained increase is not treated as an unexplained anomaly.

An unexplained increase is flagged for review.

---

# 2. High-Level Architecture

```text
                     Shipment Records
                           |
                           v
                  Data Preprocessing
                           |
                           v
                 Weekly Route Metrics
                           |
                           v
                  Anomaly Detection
                           |
                 +---------+---------+
                 |                   |
              Normal              Anomaly
                 |                   |
                 v                   v
          No RAG / LLM          Context Retrieval
                                     |
                                     v
                              Qdrant Vector DB
                                     |
                                     v
                              Retrieved Notes
                                     |
                                     v
                              LLM Validation
                                     |
                                     v
                         Deterministic Validation
                                     |
                          +----------+----------+
                          |                     |
                     Valid Evidence        No Evidence
                          |                     |
                          v                     v
                    flagged = No          flagged = Yes

freight-tiger-smart-cost-assistant/
│
├── data/
│   ├── shipment_records.csv
│   ├── context_notes.csv
│   ├── sample_output_format_v2.csv
│   └── eval_evidence_golden.csv
│
├── src/
│   └── freight_tiger_smart_cost_assistant/
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── weekly_metrics.py
│       ├── anomaly_Detection.py
│       ├── retrieval.py
│       ├── evidence_validation.py
│       └── pipeline.py
│
├── tests/
│   └── eval_evidence.py
│
├── output/
│   └── output.csv
│
├── logs/
│   └── evidence_eval_results.csv 
│
├── run.py
├── index_notes.py
├── pyproject.toml
├── uv.lock
└── README.md
```
## Code Files

### `preprocessing.py`
Loads and cleans the shipment data.

- Validates required columns
- Converts dates and numeric values
- Removes invalid records
- Assigns each shipment to a Monday-Sunday week

### `weekly_metrics.py`
Calculates weekly freight costs for each route.

- Groups shipments by route, route type and week
- Calculates total freight cost
- Calculates total tonne-km
- Calculates cost per tonne-km

### `anomaly_Detection.py`
Identifies unusual route-week costs.

- Calculates the previous 8-week average for each route
- Calculates the same-week average of similar routes
- Calculates percentage differences
- Marks anomaly candidates using the configured threshold

### `retrieval.py`
Handles RAG-based context retrieval.

- Loads context notes
- Creates embeddings using Nomic
- Stores/searches notes in Qdrant
- Filters notes by route/scope
- Retrieves relevant notes for an anomalous route-week

### `evidence_validation.py`
Checks whether retrieved notes genuinely explain a cost increase.

- Sends retrieved notes to the LLM
- Checks route and time applicability
- Checks whether the note actually explains a cost increase
- Prevents unsupported note IDs from being accepted
- Fails closed if evidence cannot be validated

### `pipeline.py`
Connects all components into the complete workflow.

- Runs preprocessing
- Calculates weekly metrics
- Detects anomalies
- Retrieves context for anomalies
- Validates evidence using the LLM
- Produces the final `flagged` decision
- Saves the required output CSV

### `index_notes.py`
Creates the Qdrant vector index from the context notes.

### `run.py`
Entry point for running the complete application.

```bash
uv run python run.py

Installation

The project uses uv.

Create the environment:

uv venv

Activate it on Windows:

.venv\Scripts\activate

Install dependencies:

uv sync

If pytest needs to be installed separately:

uv add --dev pytest
Environment Variables

Create a .env file:

GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key

Do not commit .env to version control.

Add:

.env

to .gitignore.

 Start Qdrant

Qdrant is used as the vector database.

Start the Qdrant Docker container:

docker run -p 6333:6333 qdrant/qdrant

The application connects to:

http://localhost:6333
 Index Context Notes

Before running the pipeline, index the context notes:

uv run python index_notes.py
```
This creates the Qdrant collection containing the embedded context notes.
## Testing & Evaluation

###  Golden Dataset Evaluation
Golden Set Design

The golden set contains positive and negative evidence cases.

Examples include:

Exact route + exact week
Multi-week events
Nationwide events
Events that have already ended
Notes that explicitly say costs were not affected
Notes describing normal conditions
Cases where no supporting note exists

The labels are manually determined from the supplied context notes rather
than generated by another LLM.

This avoids using an LLM to create the answer key being used to evaluate
another LLM.
## Evaluation Metrics

The RAG + LLM evidence-validation layer is evaluated against the manually
labeled golden dataset using the following metrics:

1. **Support Decision Accuracy**  
   Measures how often the system correctly determines whether a valid
   supporting context note exists.

2. **Citation Accuracy**  
   Measures how often the system identifies the correct supporting note ID
   when evidence exists.

3. **Overall Case Accuracy**  
   A case is considered correct only when both the evidence decision and
   cited note ID match the expected result.

4. **False Justification Rate**  
   Measures how often the system incorrectly accepts evidence when no valid
   explanation exists. Lower is better; ideally this should be 0%.

5. **False Negative Rate**  
   Measures how often the system fails to recognize valid supporting evidence.
   Lower is better.
## Reproducibility

The deterministic parts of the pipeline produce consistent results across runs.

Key deterministic components:
- Weekly grouping and cost calculations
- Historical and similar-route baselines
- Anomaly thresholding
- Route filtering and citation validation

The LLM uses `temperature = 0`.

The following fields should remain consistent across runs:
- `route`
- `week_of`
- `cost_per_tonne_km`
- `vs_own_history`
- `vs_similar_routes`
- `flagged`
- `matched_note_id`

The generated `reason` text may vary slightly.

---

## Cost Tracking

LLM API usage is the main variable cost.

Local components have no API cost:
- Pandas calculations
- Nomic embeddings
- Qdrant

The system tracks:
- LLM calls
- Input tokens
- Output tokens
- Estimated cost

Cost is calculated as:

```text
cost =
(input_tokens / 1,000,000 × input_price)
+
(output_tokens / 1,000,000 × output_price)
```
## Reliability Strategy

The system does not rely on a single LLM response. Reliability is achieved through multiple layers:

### Layer 1 — Deterministic Numerical Calculations
Cost calculations, baselines, and anomaly detection are implemented in code.

### Layer 2 — Metadata-Constrained Retrieval
Only relevant route/scope notes are retrieved.

### Layer 3 — Semantic Retrieval
Relevant context is selected using embeddings.

### Layer 4 — Strict Evidence Prompt
The LLM is explicitly instructed not to invent facts or causal explanations.

### Layer 5 — Deterministic Output Validation
The cited note ID must exist in the retrieved context.

### Layer 6 — Fail-Closed Behavior
If evidence cannot be validated, the anomaly remains flagged.

### Layer 7 — Golden-Set Evaluation
The evidence-validation layer is evaluated against manually labeled cases.

> The goal is not to claim 100% reliability, but to make failures constrained, detectable, and measurable.


## Assumptions

1. `origin + destination` uniquely identifies a route.
2. The provided `route_type` values are trusted and used directly.
3. Shipment quantities and distances are positive after preprocessing.
4. Weeks are Monday-Sunday.
5. The own-history baseline uses up to the previous 8 available route-week observations.
6. Missing historical weeks are not artificially padded.
7. Similar routes are defined as routes sharing the same `route_type` in the same week.
8. Context notes are considered valid evidence only when they apply to the route/scope and requested time period.
9. Notes describing normal or stable conditions are not considered explanations for a cost increase.
10. The anomaly threshold is configurable, with 10% used as the current engineering default.

---

## Known Limitations

### Temporal Reasoning

Context notes contain a publication/date field rather than explicit `valid_from` and `valid_to` fields.

Temporal applicability is therefore interpreted from the natural-language note.

A production system could improve this by storing:

```text
valid_from
valid_to
```
### LLM Dependency

Evidence validation depends on an external LLM provider.

API rate limits, quotas, or outages can affect the evidence-validation stage. The system uses fallback providers and fail-closed behavior.

### Small Context Corpus

The supplied context-note dataset is small.

For a production system with thousands of notes, additional retrieval optimization and indexing strategies would be useful.

### Threshold Selection

The assignment does not prescribe a numerical anomaly threshold.

The current 10% threshold is an engineering choice and is configurable.

A production system could learn thresholds from historical distributions, route categories, or business-defined risk tolerances.

### Natural-Language Questions

Supporting arbitrary natural-language questions is a stretch goal and is not required for the core pipeline.

The current implementation focuses on reliable route-week cost monitoring and evidence validation.
