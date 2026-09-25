# CRAG VLM Research Assistant

A self-correcting Retrieval-Augmented Generation (RAG) system built on **LangGraph**, implementing the **Corrective RAG (CRAG)** pattern from [Yan et al., 2024](https://arxiv.org/abs/2401.15884). The system answers questions over a corpus of vision-language model (VLM) research papers, grading its own retrieval quality and falling back to live web search when local context is insufficient or incomplete.

## Why CRAG

Standard RAG trusts retrieval blindly: whatever the vector store returns becomes the context for generation, with no check on whether it's actually good. CRAG adds a correction layer — a lightweight grader evaluates each retrieved chunk, and the system routes to one of three paths based on that evaluation:

- **Correct** — retrieved context is sufficient, use it as-is
- **Ambiguous** — context is partially relevant, augment it with live web search
- **Incorrect** — context is not relevant at all, discard it and rely on web search

This project implements that pattern end-to-end, including query rewriting for the web-fallback path, a hallucination/grounding check on generated answers, and a bounded retry loop.

## Architecture

```
retrieve → grade_documents → [route] → merge_context → generate → check_hallucination
                                 │            ↑                         │
                          correct│      rewrite_query                   │
                                 │            ↓                    [grounded?]
                       ambiguous/incorrect → web_search        retry ↩──┘ (capped)
```

**State schema** (`src/graph/state.py`): a single `CRAGState` TypedDict carries the question, retrieved/graded chunks, web results, the routing decision, merged context, the generated answer, grounding status, and a retry counter through every node.

**Nodes** (`src/graph/nodes.py`):
| Node | Responsibility |
|---|---|
| `retrieve` | Embed the question (BGE, asymmetric query prefix) and query Chroma for top-k chunks |
| `grade_documents` | Batched LLM call grading every chunk's relevance **and** whether the retrieved set as a whole covers the full question |
| `rewrite_query` | Rewrites the question into a web-search-optimized query |
| `web_search` | Tavily search, results normalized into the same chunk shape as local retrieval |
| `merge_context` | Branches on the routing decision to build the final context (local-only / local+web / web-only) |
| `generate` | Answers strictly from the merged context, citing sources |
| `check_hallucination` | Independent LLM call verifying every claim in the answer is grounded in the context |

**Edges** (`src/graph/edges.py`): plain Python, no LLM calls — the routing decision and retry-or-end decision are both deterministic functions over state, which makes them unit-testable independent of any model.

## Corpus

~80 papers on vision-language models, spanning CLIP-based captioning, test-time adaptation, VLM adversarial robustness, and captioning evaluation metrics — pulled via the arXiv API, deliberately mixing foundational papers (CLIP, ClipCap, RLCF) with recent ones, so that genuinely out-of-corpus questions are possible without staging them artificially.

PDF extraction uses **Nougat** (Meta's vision-transformer OCR model for scientific documents) rather than plain text extraction — this was a deliberate mid-project pivot after PyMuPDF-based extraction produced two systematic bugs: section headers being missed (causing content to bleed into the wrong section label) and LaTeX tables being flattened into unusable symbol strings. Nougat's markdown output solved the first cleanly and made the second detectable (`\begin{table}` markers), after which table content is extracted separately and excluded from the embedded corpus rather than polluting retrieval with unparseable numeric fragments.

Chunking is section-aware (via markdown heading parsing), ~300–500 tokens with overlap, tagged with `source_paper`, `section_name`, `content_type`, and `chunk_index` metadata.

## Embedding model: three iterations

| Model | Result |
|---|---|
| `all-MiniLM-L6-v2` (general-purpose) | Baseline. Failed to retrieve two known-present papers (VLRM, BEiTScore) even at k=10; compound/comparison questions retrieved neither expected source. |
| `sentence-transformers/allenai-specter` | Domain-tuned for *document-to-document* similarity (citation prediction) — wrong task shape for query→passage retrieval. Made no improvement on the same failing cases; confirmed via direct inspection that the target content existed in the corpus and was well-formed, ruling out a chunking bug. |
| `BAAI/bge-small-en-v1.5` (**current**) | Purpose-built for asymmetric query-to-passage retrieval. Fixed both known failures (VLRM and BEiTScore both became rank-1 hits) and fixed retrieval for all three compound/comparison questions in the eval set. Requires a query-side instruction prefix (`"Represent this sentence for searching relevant passages: "`) not applied to documents — an easy detail to miss that materially affects results. |

**Lesson**: a domain-tuned embedding model isn't automatically better than a general one if it's tuned for the wrong *task shape*. Matching the model's training objective (asymmetric retrieval) mattered more than matching its training domain (scientific text).

## LLM selection: five configurations tested

Grading and generation both went through multiple model swaps, driven by real failures at each stage — summarized here since the trade-offs are informative in their own right, not just the final pick.

| Config | Grading behavior | Notes |
|---|---|---|
| Local `llama3.2:3b` | Too lenient — graded clearly out-of-scope content (a GPU-systems paper) as relevant | Rejected: risks false "correct" routing on genuinely irrelevant retrieval |
| Local `qwen2.5:7b-instruct` | Fixed false positives, but over-strict — penalized supporting/background chunks that didn't use the question's exact wording | Routing skewed correct→ambiguous, but answer quality was unaffected since the ambiguous path augments rather than discards context |
| Local `qwen3:8b` | Similar profile to Qwen2.5, slightly better | Hit a genuine hardware ceiling (100% GPU utilization, ~4.5/6GB VRAM on a GTX 1660 Ti) — full 14-question eval took **128 minutes** |
| Hosted Gemini Flash/Pro | Fast when available | Free-tier rate limits (5 req/min) and transient 503 "model overloaded" errors caused repeated pipeline failures; required a retry-with-backoff wrapper and, even then, unreliable for iterative development |
| **Groq (`gpt-oss-20b` grading / `gpt-oss-120b` generation), current** | Fast (LPU hardware) and free at project scale (14,400 req/day) | Full eval dropped to **7.8 minutes**. See below for the accuracy trade-off this introduced. |

### A genuine, counterintuitive finding: bigger grading model ≠ better routing accuracy

| Grading model | Correct-bucket accuracy | Ambiguous-bucket accuracy | Incorrect-bucket accuracy |
|---|---|---|---|
| Local Qwen3:8b | 2/7 (29%) | 3/3 (100%) | 4/4 (100%) |
| Groq gpt-oss-20b | 4/7 (57%) | 1/3 (33%) | 4/4 (100%) |
| Groq gpt-oss-120b | 5/7 (71%) | 0/3 (0%) | 4/4 (100%) |

As the grading model got larger, single-topic ("correct" bucket) routing accuracy improved — but comparison-question ("ambiguous" bucket) coverage detection got *worse*, collapsing to 0% with the largest model. The likely explanation: larger instruction-tuned models are more confident and more willing to synthesize a "yes, this covers it" judgment even from one-sided evidence, while smaller models hedge toward "no" more readily — which happens to align with the correct answer on comparison questions, likely more by way of general caution than genuine discernment.

**Incorrect-bucket routing was 100% accurate across every single configuration tested** — regardless of model size or provider, the system reliably recognized genuinely out-of-corpus questions and triggered web fallback. This is the most safety-critical behavior in the pipeline and it never wavered.

**Final choice**: the hybrid config (20b grading / 120b generation) — matches the all-120b config's overall accuracy at roughly 30% less runtime and lower API usage.

## Routing logic: threshold, not unanimity

The initial design required 100% of retrieved chunks to be graded relevant before routing "correct." In practice this proved too strict — a single conservatively-graded supporting chunk was enough to tip a well-covered question into "ambiguous." The aggregation now uses a majority threshold (`CORRECT_THRESHOLD = 0.6`), closer to the original CRAG paper's framing of an overall confidence judgment rather than literal per-chunk unanimity.

A second, separate mechanism catches what the threshold change couldn't: **compound/comparison questions** (e.g. "how does X compare to Y") where every individual chunk can be legitimately "relevant" to *one side* of the comparison, while the retrieved set as a whole is still incomplete. The grading call returns a set-level `fully_covers_question` boolean alongside per-chunk grades — if every chunk is relevant but coverage is incomplete, the routing decision is downgraded from "correct" to "ambiguous" to trigger web augmentation. This is the mechanism whose reliability varies by grading model size, as shown above.

## Known limitations (deliberate scope decisions, not oversights)

- **The hallucination-check retry loop re-attempts generation with the same context, not a fresh retrieval cycle.** A persistent `not_grounded` verdict typically means the underlying context was genuinely insufficient — retrying `generate` with identical input rarely fixes that. This was a deliberate simplification: extending the retry to loop back through retrieval was considered and rejected, since `check_hallucination` only observes the final answer and has no way to diagnose *why* grounding failed (retrieval miss vs. grader miscalibration vs. a genuinely unanswerable question) — an untargeted "try web search again" fallback isn't guaranteed to help and adds complexity without a clear mechanism.
- **Table content is extracted but excluded from the embedded corpus.** Even with Nougat's LaTeX-aware extraction, flattened table cell values are largely unusable as retrieval context. Tables are detected and logged separately in Stage 2 output for future work rather than silently discarded.
- **The grading model's coverage-check reliability is model-dependent**, as documented above — this is presented as an empirical finding, not swept under the rug.
- **The generation model is instructed to refuse rather than hallucinate** when context is insufficient — observed directly during testing (a deliberately mismatched-context test produced an honest "I cannot answer from this context" rather than a confabulated response). This is a positive robustness property, but means the retry loop is rarely exercised in practice, since the model's default behavior is already conservative.

## Eval methodology

14 hand-written questions across three buckets (`eval_set.json`), each with an expected routing outcome and answer-content notes, built directly from manual inspection of the actual corpus (not synthetic/generic questions):
- **Correct** (7): directly answerable from a single known-present paper
- **Ambiguous** (3): comparison questions across two papers with uneven corpus coverage
- **Incorrect** (4): genuinely out-of-corpus (different subfield entirely, or post-cutoff content)

`run_eval.py` checks retrieval-only quality (chunk-level hit/miss against expected source papers); `test_checkpoint7.py` runs the full compiled graph end-to-end and compares actual vs. expected routing, plus logs generation quality, hallucination status, retry counts, and per-question latency.

## Tech stack

- **Orchestration**: LangGraph (StateGraph, conditional edges)
- **LLMs**: Groq-hosted `openai/gpt-oss-20b` (grading, query rewriting) and `openai/gpt-oss-120b` (generation, hallucination check), via `langchain_openai.ChatOpenAI` pointed at Groq's OpenAI-compatible endpoint
- **Embeddings**: `BAAI/bge-small-en-v1.5` (sentence-transformers)
- **Vector store**: ChromaDB (persistent, local)
- **Web fallback**: Tavily
- **PDF extraction**: Nougat (Meta)

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in:
```
GROQ_API_KEY=
TAVILY_API_KEY=
```

Ingest the corpus (fetches papers, extracts via Nougat, chunks, embeds into Chroma):
```bash
python src/ingestion/arxiv_fetch.py
python src/ingestion/pdf_extract.py
python src/vectorstore/rebuild_chroma_bge.py
```

Run the full eval:
```bash
python test_checkpoint7.py
```

## Project structure

```
├── data/                    # corpus artifacts (gitignored)
│   ├── raw_pdfs/              # downloaded papers + manifest.json
│   └── processed/              # Nougat markdown + chunked JSON
├── src/
│   ├── ingestion/                # arXiv fetch, PDF extraction, chunking
│   ├── vectorstore/                # Chroma population scripts
│   └── graph/                       # LangGraph implementation
│       ├── state.py
│       ├── nodes.py
│       ├── edges.py
│       ├── config.py
│       └── build_graph.py
├── eval_set.json              # ground-truth eval questions
└── test_checkpoint*.py        # incremental validation scripts (1-7)
```

## Future work

- Extend the retry loop to distinguish retrieval-miss, grader-miscalibration, and genuinely-unanswerable failure modes via richer logging, rather than a single undifferentiated `not_grounded` signal
- Table/figure extraction into a queryable form, rather than exclusion
- Multi-provider generation benchmarking (the original project scope listed OpenAI/DeepSeek/Llama/Claude as candidates) using the eval harness already in place