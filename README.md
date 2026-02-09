# C2RAG

This repository contains the reference implementation of the paper:

> **Mitigating KG Quality Issues: A Robust Multi-Hop GraphRAG Retrieval Framework** (ICML 2026 submission)

C2RAG targets **robust multi-hop question answering over imperfect knowledge graphs (KGs)**. It is designed to reduce two common failure modes in multi-hop graph retrieval:

- **Retrieval drift**: noisy/spurious KG edges gradually divert the retrieval trajectory.
- **Retrieval hallucination**: missing KG evidence causes the retriever to keep propagating structurally without sufficient support.
  
![Architecture of C2RAG](framework)

## Repository Structure

The code is organized to mirror the paper modules.

- `configs/`
  - `config.py`: CLI arguments and dataset-specific default paths.
  - `io.py`: JSON helpers and query-plan normalization.
- `datasets/`
  - `dataset/`: benchmark QA files (`2wiki`, `hotpotqa`, `musique`).
  - `kg_*_1000_json/`: prebuilt KG bundles aligned to the benchmarks.
- `kg/`: KG loading + indexing utilities (entity list, triple list, 1-hop indices).
- `modules/`: core method modules
  - `Atomic_Constraint_Planning.py`: atomic constraint planning (query decomposition).
  - `anchor_matching.py`: anchor entity matching from constraints to KG entities.
  - `relation_alignment.py`: relation semantic alignment (embedding similarity; query uses `relation_variants`).
  - `contextual_reranking.py`: cross-encoder reranking with question + constraint context.
  - `sufficiency_check.py`: hop-wise sufficiency check using effective-support (`N_eff`).
  - `binding_propagation.py`: variable binding consolidation + binding-consistent filtering/propagation.
  - `textual_recovery.py`: fallback textual recovery for unresolved constraints.
  - `dataset_adapters.py`, `provenance.py`, `query_texts.py`: dataset adapters and utilities.
- `pipelines/`
  - `pipeline1.py`: Stage-1 structural retrieval orchestration.
  - `pipeline2.py`: Stage-2 textual recovery + evidence assembly.
- `rerank/`: cross-encoder wrapper (HF `AutoModelForSequenceClassification`).
- `sim/`: dense embedding wrapper.
- `scripts/`: runnable entry points
  - `run_stage1.py`, `run_stage2.py`, `run.py`: Stage-1, Stage-2, or end-to-end.
  - `evaluation.py`: QA EM/F1 + retrieval recall evaluation.

---

## Requirements

- Python >= 3.10
- Core packages:
  - `torch`
  - `transformers`
  - `sentence-transformers` (recommended; used for dense similarity)
  - `openai` (for Atomic Constraint Planning and evaluation-time QA generation)
  - `tqdm`, `numpy`

Example (CPU-only):

```bash
pip install torch transformers sentence-transformers openai tqdm numpy
```

If you have a CUDA GPU, `torch` + `transformers` will automatically leverage it for the reranker when available.

---

## Data and KG Format

This repo includes:

- QA benchmarks: `datasets/dataset/`
  - `2wikimultihopqa.json`
  - `hotpotqa.json`
  - `musique.json`
- Prebuilt KGs: `datasets/kg_*_1000_json/`
  - `entities.jsonl`: entity inventory
  - `triples.jsonl`: KG triples
  - `title2entities.jsonl`: document-to-entity mapping
  - `title2triples.jsonl`: document-to-triple mapping

To plug in your own KG, ensure `kg/loader.py` can load it (either a `.pt`, a `.json` containing `{entities, triples}`, or a directory of `.jsonl` files).

---

## Running C2RAG

### Important: run as a Python module

Several scripts use package-relative imports (e.g., `from ..configs...`). To run reliably:

1. Make sure the repository folder name is a valid Python identifier (no hyphens). For example, rename the repo root to `c2rag`.
2. Run with `python -m` from the **parent directory** so the package is importable.

Example:

```bash
# Clone into a valid package name
git clone <REPO_URL> c2rag

# Run from the parent directory
cd ..
python -m c2rag.scripts.run --dataset musique
```

### Step 0: Atomic Constraint Planning (Query Decomposition)

C2RAG expects a `query_json` containing decomposed constraint triples with the schema:

```json
{
  "id": "...",
  "question": "...",
  "ground_truth_answer": "...",
  "query_plan": [
    {"head": "...", "tail": "...", "relation_variants": ["...", "...", "..."]}
  ]
}
```

Notes:
- Query-side canonical `relation` is **not required**. The pipeline uses **`relation_variants`** for semantic alignment.
- Variables should be `?`-prefixed (e.g., `?person`, `?country`).

Generate `query_json` programmatically (writes to `args.query_json` from `configs/config.py`):

```bash
cd ..
python - <<'PY'
from c2rag.configs.config import parse_args
from c2rag.modules.Atomic_Constraint_Planning import main_query

args = parse_args()
main_query(args)
PY
```

You can override planner endpoint settings with:
- `--api_key`, `--base_url`, `--model_name`

### 1: Structural Retrieval (Constraint-based)+Sufficiency Check+Binding Propagation

Stage-1 consumes `query_json` and retrieves KG triples per constraint.

```bash
cd ..
python -m c2rag.scripts.run_stage1 --dataset musique
```

implements:
- **Anchor Matching**: match constraint entities to KG entities (`modules/anchor_matching.py`)
- **Relation Alignment**: filter candidates by relation similarity (`modules/relation_alignment.py`)
- **Contextual Reranking**: rerank candidates with a cross-encoder (`modules/contextual_reranking.py`)
- **Sufficiency Check**: decide resolved/unresolved via `N_eff` (`modules/sufficiency_check.py`)
- **Binding Propagation**: consolidate variable bindings and enforce consistency (`modules/binding_propagation.py`)

Output: `stage1_json` (default: `result/<dataset>/stage1_<dataset>.json`).

### 2. Textual Recovery + Evidence Assembly

Stage-2 consumes Stage-1 outputs and builds **per-constraint evidence blocks**:

- For **resolved** constraints, evidence is mapped back to dataset context (sentence/paragraph provenance).
- For **unresolved/insufficient** constraints, Stage-2 retrieves evidence directly from raw documents.

```bash
cd ..
python -m c2rag.scripts.run_stage2 --dataset musique
```

Output: `stage2_json` (default: `result/<dataset>/stage2_<dataset>.json`).

### End-to-end (Stage-1 + Stage-2)

```bash
cd ..
python -m c2rag.scripts.run --dataset musique
```

---

## Evaluation

The evaluation script uses Stage-2 evidence to generate an answer (via an OpenAI-compatible chat endpoint) and reports:

- **QA EM** and **QA F1** (against gold answers)
- **Retrieval recall@K** against dataset supports
  - `2wiki`/`hotpotqa`: sentence-level supports (`(title, sent_idx)`)
  - `musique`: paragraph-level supports (`paragraph_support_idx`)

Run:

```bash
cd ..
python -m c2rag.scripts.evaluation \
  --dataset musique \
  --stage2_json result/musique/stage2_musique.json \
  --eval_output_json result/musique/eval.json
```

Common evaluation options:
- `--api_key`, `--base_url`, `--model_name`: LLM endpoint for answer generation.
- Prompt budgeting (if enabled in your config):
  - `--prompt_max_triples`
  - `--prompt_max_text_per_triple`
- Recall reporting (if enabled):
  - `--recall_k_list` (e.g., `2,5,10`)

---

## Outputs

By default (see `configs/config.py`):

- `result/<dataset>/automic_query_<dataset>.json`: decomposed atomic constraint plans
- `result/<dataset>/stage1_<dataset>.json`: selected structural KG evidence + bindings/debug metadata
- `result/<dataset>/stage2_<dataset>.json`: per-constraint evidence blocks (input to evaluation)
- `result/<dataset>/eval.json`: predictions + EM/F1 + recall (if enabled)

