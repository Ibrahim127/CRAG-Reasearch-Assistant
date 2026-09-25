# CRAG Research Assistant

A self-correcting Retrieval-Augmented Generation (RAG) system built on **LangGraph**, implementing the **Corrective RAG (CRAG)** pattern from [Yan et al., 2024](https://arxiv.org/abs/2401.15884). Point it at any set of PDFs — or search and pull papers directly from arXiv — and it answers questions over them, grading its own retrieval quality and falling back to live web search when local context is insufficient or incomplete.

Includes a Streamlit UI for uploading documents, fetching arXiv papers by search query, and chatting against the resulting knowledge base, with the routing decision and grounding status shown per answer.

## Why CRAG

Standard RAG trusts retrieval blindly: whatever the vector store returns becomes the context for generation, with no check on whether it's actually good. CRAG adds a correction layer — a lightweight grader evaluates each retrieved chunk, and the system routes to one of three paths based on that evaluation:

- **Correct** — retrieved context is sufficient, use it as-is
- **Ambiguous** — context is partially relevant, augment it with live web search
- **Incorrect** — context is not relevant at all, discard it and rely on web search

This project implements that pattern end-to-end: query rewriting for the web-fallback path, a hallucination/grounding check on generated answers, and a bounded retry loop.

## Architecture

```
retrieve → grade_documents → [route] → merge_context → generate → check_hallucination
                                 │            ↑                         │
                          correct│      rewrite_query                   │
                                 │            ↓                    [grounded?]
                       ambiguous/incorrect → web_search        retry ↩──┘ (capped)
```

**State schema** (`src/graph/state.py`): a `CRAGState` TypedDict carries the question, which document collection to query, retrieved/graded chunks, web results, the routing decision, merged context, the generated answer, grounding status, and a retry counter through every node.

**Nodes** (`src/graph/nodes.py`):
| Node | Responsibility |
|---|---|
| `retrieve` | Embed the question (BGE, asymmetric query prefix) and query the active Chroma collection for top-k chunks |
| `grade_documents` | Batched LLM call grading every chunk's relevance **and** whether the retrieved set as a whole covers the full question |
| `rewrite_query` | Rewrites the question into a web-search-optimized query |
| `web_search` | Tavily search, results normalized into the same chunk shape as local retrieval |
| `merge_context` | Branches on the routing decision to build the final context (local-only / local+web / web-only) |
| `generate` | Answers strictly from the merged context, citing sources |
| `check_hallucination` | Independent LLM call verifying every claim in the answer is grounded in the context |

**Edges** (`src/graph/edges.py`): plain Python, no LLM calls — the routing decision and retry-or-end decision are both deterministic functions over state, unit-testable independent of any model.

## Bring your own documents

The system isn't tied to a fixed corpus. `src/ingestion/ingest_documents.py` is a standalone pipeline that turns any folder of PDFs into a named, independently queryable Chroma collection:

```bash
python src/ingestion/ingest_documents.py --pdf_dir path/to/pdfs --collection_name my_docs
```

Or use the Streamlit app (`streamlit run app/app.py`), which wraps the same pipeline with a file-upload UI **and** an arXiv search tab — search by keyword, pick how many results to pull, and it downloads and ingests them directly, no manual PDF handling required.

`CHROMA_PATH` and `COLLECTION_NAME` are environment-variable-configurable (`src/graph/config.py`, defaulting sensibly if unset), so the storage location and default corpus aren't hardcoded — multiple collections can coexist, and the graph queries whichever one is active per-request via `state["collection_name"]`.

### Extraction: cloud-based, not hardware-dependent

Document parsing runs on **LlamaParse** (LlamaIndex's cloud parsing API, lite/fast mode), not a local model. This was a deliberate choice after evaluating GPU-dependent local options (Nougat, Marker): those produce excellent extraction quality but require a CUDA-capable GPU and, for Marker specifically, a heavy first-run model download — both real barriers for a "plug and play" tool meant to run on whatever machine a user has. LlamaParse removes that dependency entirely at some cost to maximum extraction fidelity, with a local `pypdf`/`pdfplumber` fallback chain if the API is unavailable or a specific document fails.

## Corpus (default)

The project ships with a default corpus of ~80 papers on vision-language models — CLIP-based captioning, test-time adaptation, VLM adversarial robustness, and captioning evaluation metrics — pulled via the arXiv API, deliberately mixing foundational papers (CLIP, ClipCap, RLCF) with recent ones so genuinely out-of-corpus eval questions are possible without staging them artificially. Chunking is section-aware (markdown heading parsing), ~300–500 tokens with overlap, tagged with `source_paper`, `section_name`, `content_type`, and `chunk_index` metadata.

## Embedding model: three iterations

| Model | Result |
|---|---|
| `all-MiniLM-L6-v2` (general-purpose) | Baseline. Failed to retrieve two known-present papers even at k=10; compound/comparison questions retrieved neither expected source. |
| `sentence-transformers/allenai-specter` | Domain-tuned for *document-to-document* similarity (citation prediction) — wrong task shape for query→passage retrieval. No improvement on the same failing cases, confirmed via direct inspection that the target content existed and was well-formed, ruling out a chunking bug. |
| `BAAI/bge-small-en-v1.5` (**current**) | Purpose-built for asymmetric query-to-passage retrieval. Fixed both known failures and all compound/comparison questions in the eval set. Requires a query-side instruction prefix (`"Represent this sentence for searching relevant passages: "`) not applied to documents. |

**Lesson**: a domain-tuned embedding model isn't automatically better than a general one if it's tuned for the wrong *task shape*. Matching the model's training objective (asymmetric retrieval) mattered more than matching its training domain (scientific text).

## LLM selection

Grading and generation went through several iterations, each driven by a concrete observed failure:

| Config | Outcome |
|---|---|
| Local `llama3.2:3b` (Ollama) | Too lenient — graded clearly out-of-scope content as relevant |
| Local `qwen2.5:7b-instruct` | Fixed false positives, but over-strict on supporting/background context |
| Local `qwen3:8b` | Best local result, but hit a genuine hardware ceiling on a 6GB VRAM GPU — full eval took 128 minutes |
| Hosted Gemini Flash/Pro | Fast when available, but free-tier rate limits and transient 503s caused repeated pipeline failures |
| **Groq (`openai/gpt-oss-20b` grading / `openai/gpt-oss-120b` generation), current** | Free tier (14,400 req/day), fast (custom LPU hardware) — full eval dropped to 7.8 minutes |

### A genuine, counterintuitive finding: bigger grading model ≠ better routing accuracy

| Grading model | Correct-bucket accuracy | Ambiguous-bucket accuracy | Incorrect-bucket accuracy |
|---|---|---|---|
| Local Qwen3:8b | 2/7 (29%) | 3/3 (100%) | 4/4 (100%) |
| Groq gpt-oss-20b | 4/7 (57%) | 1/3 (33%) | 4/4 (100%) |
| Groq gpt-oss-120b | 5/7 (71%) | 0/3 (0%) | 4/4 (100%) |

As the grading model got larger, single-topic routing accuracy improved, but comparison-question coverage detection got *worse*. Larger instruction-tuned models appear more willing to synthesize a confident "this covers it" judgment from one-sided evidence; smaller models hedge toward "no" more readily, which happens to align with the correct answer on comparison questions.

**Incorrect-bucket routing was 100% accurate across every configuration tested**, regardless of model size or provider — the most safety-critical behavior in the pipeline never wavered.

**Final choice**: the hybrid config (20b grading / 120b generation) — matches the all-120b config's overall accuracy at meaningfully less runtime and API usage.

## Routing logic: threshold, not unanimity

The initial design required 100% of retrieved chunks to be graded relevant before routing "correct." In practice this proved too strict — a single conservatively-graded supporting chunk was enough to tip a well-covered question into "ambiguous." The aggregation now uses a majority threshold (`CORRECT_THRESHOLD = 0.6`), closer to the CRAG paper's framing of an overall confidence judgment rather than literal per-chunk unanimity.

A second, separate mechanism catches what the threshold alone couldn't: **compound/comparison questions** where every individual chunk can be legitimately "relevant" to *one side* of the comparison while the retrieved set as a whole is still incomplete. The grading call returns a set-level `fully_covers_question` boolean alongside per-chunk grades — if every chunk is relevant but coverage is incomplete, the routing decision is downgraded from "correct" to "ambiguous" to trigger web augmentation. This mechanism's reliability varies by grading model size, as shown above.

## Known limitations (deliberate scope decisions)

- **The hallucination-check retry loop re-attempts generation with the same context, not a fresh retrieval cycle.** A persistent `not_grounded` verdict usually means the context was genuinely insufficient — retrying with identical input rarely fixes that. Looping back through retrieval instead was considered and rejected: `check_hallucination` only observes the final answer, with no way to diagnose *why* grounding failed (retrieval miss vs. grader miscalibration vs. a genuinely unanswerable question).
- **Table content is detected and excluded from the embedded corpus**, not silently discarded — flattened table cell values are largely unusable as retrieval context even when extracted cleanly.
- **The grading model's coverage-check reliability is model-dependent**, documented above as an empirical finding rather than a hidden inconsistency.
- **The generation model defaults to refusing rather than hallucinating** when context is insufficient, observed directly during testing — a positive robustness property that also means the retry loop is rarely exercised in practice.
- **LlamaParse's lite/fast mode trades some extraction fidelity for speed and zero hardware requirements** — a deliberate accessibility choice for a "bring your own documents" tool, with a local fallback chain if the API is unavailable.

## Eval methodology

14 hand-written questions across three buckets (`eval_set.json`), each with an expected routing outcome and answer-content notes, built from manual inspection of the actual default corpus:
- **Correct** (7): directly answerable from a single known-present paper
- **Ambiguous** (3): comparison questions across two papers with uneven corpus coverage
- **Incorrect** (4): genuinely out-of-corpus (different subfield entirely, or post-cutoff content)

`run_eval.py` checks retrieval-only quality; `test_checkpoint7.py` runs the full compiled graph end-to-end and compares actual vs. expected routing, plus logs generation quality, hallucination status, retry counts, and per-question latency.

## Tech stack

- **Orchestration**: LangGraph (StateGraph, conditional edges)
- **LLMs**: Groq-hosted `openai/gpt-oss-20b` (grading, query rewriting) and `openai/gpt-oss-120b` (generation, hallucination check), via `langchain_openai.ChatOpenAI` pointed at Groq's OpenAI-compatible endpoint
- **Embeddings**: `BAAI/bge-small-en-v1.5` (sentence-transformers)
- **Vector store**: ChromaDB (persistent, local)
- **Document parsing**: LlamaParse (cloud), with local `pypdf`/`pdfplumber` fallback
- **Web fallback**: Tavily
- **UI**: Streamlit

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in:
```
GROQ_API_KEY=
TAVILY_API_KEY=
LLAMA_CLOUD_API_KEY=
```

Ingest documents:
```bash
python src/ingestion/ingest_documents.py --pdf_dir path/to/pdfs --collection_name my_docs
```

Or launch the UI:
```bash
streamlit run app/app.py
```

Run the full eval against the default corpus:
```bash
python test_checkpoint7.py
```

## Project structure

```
├── data/                    # corpus artifacts (gitignored)
├── src/
│   ├── ingestion/              # generic PDF ingestion pipeline (LlamaParse)
│   ├── vectorstore/              # Chroma population scripts
│   └── graph/                     # LangGraph implementation
│       ├── state.py
│       ├── nodes.py
│       ├── edges.py
│       ├── config.py
│       └── build_graph.py
├── app/
│   └── app.py                # Streamlit UI: upload / arXiv search / chat
├── eval_set.json              # ground-truth eval questions
└── test_checkpoint*.py        # incremental validation scripts (1-7)
```

## Future work

- Extend the retry loop to distinguish retrieval-miss, grader-miscalibration, and genuinely-unanswerable failure modes via richer logging
- Optional higher-fidelity extraction path (Nougat/Marker) for users with GPU access who want maximum quality on dense academic PDFs
- Multi-provider generation benchmarking using the eval harness already in place
