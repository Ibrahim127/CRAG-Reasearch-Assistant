# Corrective RAG (CRAG) Stateful Workflow Project
dont we lose so much info?Weighed information loss against user needs and context preservationWeighed information loss against user needs and context preservationGood instinct to question this — let's be concrete about what's actually lost at each step, because some of it matters and some of it doesn't.
What you genuinely lose
At PDF → Stage 2 (text extraction):

Figures, charts, tables, images — pure text extraction (PyMuPDF's basic mode) pulls only text, not visual content. If a paper's key result is in a chart or a results table, plain text extraction either mangles it (tables often extract as garbled fragments) or drops it entirely.
Layout/formatting meaning — bold, italics, equations rendered as images, footnotes — mostly flattened into plain text, losing whatever emphasis or structure they conveyed.
References section — you're deliberately stripping this, on purpose (it's noise for retrieval, not signal)
Precise page/figure numbers — if a chunk says "as shown in Figure 3," that figure is gone, so that reference becomes meaningless to the LLM

At Stage 2 → Stage 3 (chunking):

Cross-chunk context — a chunk in isolation loses the surrounding paragraph's context. If sentence 1 defines a term and sentence 40 (two chunks later) uses it, the retrieval unit that gets the term-use sentence may not carry the definition with it.
Document-level narrative — you lose the sense of "this is building an argument across the whole paper" — each chunk is retrieved independently, so the LLM sees fragments, not flow.

What mitigates each of these (and how much it actually matters here)

Chunk overlap (e.g. 50-100 token overlap between consecutive chunks) directly addresses the "term used without its definition" problem — some shared context carries across chunk boundaries.
Metadata per chunk (paper title, section, arxiv_id) partially compensates for lost document-level context — the LLM at least knows which paper, which section a chunk came from, and generation prompts can include this so the model doesn't treat a chunk as fully isolated.
Tables/figures — genuinely a real gap for a v1 project. If your subfield's papers lean heavily on quantitative results tables (benchmarks, ablations), you're losing real signal. This is a known, honest limitation you should write into your README rather than hide — "current version extracts body text only; table/figure extraction (e.g. via unstructured or vision-based parsing) is a natural extension" is a legitimate, professional thing to say, and actually makes your project look more thought-through, not less.
References being dropped is fine — that's intentional and not a real loss; nobody's asking the assistant to retrieve a citation list verbatim.

The actual tradeoff you're making
RAG systems inherently trade "the model sees everything, perfectly" for "the model sees the most relevant few pieces, imperfectly extracted." This isn't a flaw specific to your pipeline — it's the fundamental nature of retrieval-based systems, and it's why the grading/correction layer (CRAG's whole point) exists: because retrieval is lossy and imperfect, you build a system that can detect when what it got isn't good enough and compensate (web fallback).
So in a way — the fact that you lose info in chunking is precisely the justification for building CRAG instead of plain RAG. If retrieval were lossless and perfect, you wouldn't need a correction mechanism at all.
Practical takeaway for scope: for a v1 CV project, text-only extraction is the standard, accepted approach — nobody expects you to solve figure/table extraction in a portfolio project. Just be explicit about the limitation in your README rather than pretending it doesn't exist. If you want to go further later, that's a legitimate "future work" bullet point, not a blocker now.








INGESTION
 Document Storage Design (this is the part that matters most for next steps)
Think of this in three storage stages, each with a clear boundary:
Stage 1 — Raw storage (data/raw_pdfs/)
Just the downloaded PDFs, named predictably:
{arxiv_id}.pdf
e.g. 2103.00020.pdf. Don't rename by title — titles have slashes, colons, unicode, all of which cause pain. arXiv ID is a clean, unique, stable filename.
Alongside this, keep a manifest file — a single JSON or CSV (data/raw_pdfs/manifest.json) listing every paper you pulled, with fields: arxiv_id, title, authors, published, categories, pdf_url, abstract. This comes directly from the arXiv API response — write it once at fetch time, before you even touch PDF extraction. This manifest becomes your source of truth for "what's in my corpus" and is genuinely useful in your README ("corpus of 78 papers, list in manifest.json").
Stage 2 — Processed/intermediate storage (data/processed/)
After PDF text extraction + section detection, before chunking/embedding — store this as an intermediate artifact rather than piping straight PDF→vectorstore. Why: if your chunking strategy changes later (it will, once you see bad retrieval), you don't want to re-extract every PDF from scratch.
One JSON per paper:
data/processed/{arxiv_id}.json
{
  "arxiv_id": ...,
  "title": ...,
  "published": ...,
  "sections": [
    {"section_name": "Introduction", "text": "..."},
    {"section_name": "Method", "text": "..."},
    ...
  ]
}
This is the layer where your reference-stripping and section-detection logic lands — inspect a handful of these by hand before moving on. If section detection is garbage on a few papers, you'll see it here, cheaply, before it's buried inside embeddings.
Stage 3 — Chunked + embedded storage (the vector DB itself)
This is what actually goes into Chroma. Each chunk's metadata should carry everything you'll need for grading, citation, and later filtering — decided back in Phase B discussion:
metadata: {
  arxiv_id, title, section, published
}
Chroma persists to disk (VECTORSTORE_PATH in your config) — treat this as a build artifact, not something you hand-edit. If chunking strategy changes, you regenerate Stage 3 from Stage 2, not from raw PDFs again.
Why three stages instead of PDF → vectorstore directly: each stage is independently inspectable and independently regenerable. This is a good thing to be able to say in an interview — "I decoupled extraction from chunking from embedding so I could iterate on chunking strategy without re-running PDF parsing every time."



Repo structure

crag-research-assistant/
├── README.md                  # fill in properly at the end, stub now
├── .env.example                # keys needed, no real values
├── pyproject.toml / requirements.txt
├── config.py                   # model names, chunk size, k, thresholds — centralized
├── data/
│   ├── raw_pdfs/                # downloaded arxiv papers
│   └── processed/               # extracted/chunked text before embedding
├── src/
│   ├── ingestion/
│   │   ├── arxiv_fetch.py
│   │   ├── pdf_extract.py
│   │   └── chunker.py
│   ├── vectorstore/
│   │   └── chroma_setup.py
│   ├── graph/
│   │   ├── state.py             # your CRAGState schema
│   │   ├── nodes.py
│   │   ├── edges.py
│   │   └── build_graph.py
│   └── eval/
│       ├── eval_set.json        # your ground-truth questions
│       └── run_eval.py
├── notebooks/                   # scratch/exploration, not final code
└── app/                         # optional Streamlit/Gradio UI, later






Step-by-Step Project Plan
Phase 0 — Setup & scoping (day 1)

Pick final subfield: RAG/retrieval papers
Set up repo structure, env, API keys (LLM provider, Tavily/SerpAPI)
Decide models: cheap/fast model for grading + hallucination check, stronger model for generation and query rewriting

Phase 1 — Corpus build (days 2-3)

Pull ~40-50 foundational + ~30 recent papers via arXiv API (as scoped above)
Extract text (PyMuPDF), strip references, detect sections
Chunk section-aware, ~300-500 tokens with overlap
Embed and load into Chroma with metadata (arxiv_id, title, section, published)
Checkpoint: manually query the vector store directly (no LLM yet) for 5-10 known questions — confirm relevant chunks actually come back. Don't proceed until this looks right; garbage retrieval poisons every downstream grading decision.

Phase 2 — Baseline RAG (day 4)

Build a plain retrieve → generate chain, no grading, no LangGraph yet
Run your eval question set (build this now if you haven't) through it
Checkpoint: note where plain RAG already fails — these are your best demo cases for showing CRAG's value later

Phase 3 — Eval set finalization (day 4-5)

Lock in ~20-30 questions across the three buckets (correct/ambiguous/incorrect-expected)
Write down expected routing + a rough expected answer for each — this is your ground truth file, keep it as a JSON/CSV, not just in your head

Phase 4 — LangGraph skeleton (day 5-6)

Define the State schema (below)
Wire up nodes as no-op passthroughs first just to get the graph compiling and routing correctly
Confirm conditional edges fire on dummy/hardcoded grades before plugging in real LLM calls

Phase 5 — Grading node (day 6-7)

Implement structured-output grading prompt (per-chunk)
Implement the aggregation function (plain Python) that turns a list of grades into correct/incorrect/ambiguous
Test in isolation: feed it known-good and known-bad chunks manually, confirm grades match your expectations before wiring into the graph

Phase 6 — Web fallback + query rewriting (day 7-8)

Implement query rewrite node (only invoked on ambiguous/incorrect path)
Wire Tavily/SerpAPI call, normalize results into the same Doc shape as vector store chunks
Implement merge logic for the ambiguous path (keep relevant vector chunks + web results) vs incorrect path (discard vector chunks entirely)

Phase 7 — Generation + hallucination check (day 8-9)

Generation node using final_context
Hallucination grader: checks generation claims against final_context
Wire retry loop with retry_count cap (max 2)

Phase 8 — Full eval run (day 9-10)

Run entire eval set through the full graph
Record: routing decision vs expected, final answer quality, retry counts, latency per node
Fix misroutes by tuning grading prompt, not by hacking the aggregation thresholds

Phase 9 — Polish for CV (day 10-12)

README with the graph diagram, eval results table, example transcripts (showing all 3 routing paths)
Optional: simple Streamlit/Gradio UI showing the routing decision + sources per answer (huge for demo value — lets you show the correction happening, not just describe it)




used nougatprocessor / facebook to process pdfs into markdown sections instead of rigid regex processing
because the results from regex were so bad

can you explain what is nougat is diong diferent to regex

The fundamental difference is that Regex processes raw, extracted text rules blindly, while Nougat processes visual image pixels contextually.

When you parse a scientific PDF using standard tools + Regex, you are doing a two-step blind extraction:

An underlying tool (like PyPDF or PDFMiner) tries to extract character strings and their baseline bounding boxes from the PDF byte stream.

Your Regex rules run over that raw text stream, using pattern-matching (like r"^\d+\.\s+[A-Z]" or r"^Abstract") to guess where sections start and end.

Here is why that approach breaks, and what Meta's Nougat does differently:

1. Visual Geometry vs. Pure Text Strings
Regex: Doesn't know what a page looks like. If a paper has a two-column layout, standard text extractors often read across both columns line-by-line instead of down the left column first. Regex gets a scrambled mess where paragraph text from Column 1 merges with captions or equations from Column 2.

Nougat: Uses a Swin Transformer as a Vision Encoder. It views the page as a high-resolution image, "seeing" margins, font sizes, bold headers, and two-column layouts the same way a human reader does.

2. Contextual Understanding vs. Hardcoded Rules
Regex: Relies on rigid syntax. If one paper uses 1. Introduction, another uses I. INTRODUCTION, and a third uses Introduction without a number, your Regex misses it or misidentifies a sentence like "In this introduction, we present..." as a new section.

Nougat: Uses an mBART Text Decoder paired with its vision model. It generates Markdown semantics sequentially (# Introduction, ## Methodology) based on reading the document in context, regardless of how the author formatted the specific title style.

3. LaTeX Math Equations & Tables
Regex: Fails completely on complex LaTeX formulas. Math symbols in PDFs often extract as gibberish, broken Unicode characters, or disjointed numbers spread across multiple lines.

Nougat: Was trained explicitly on millions of scientific paper pages to translate visual mathematical formulas directly into structured, valid LaTeX code (e.g., \frac{a}{b}) right inside the text flow.

Comparison Summary
Feature	Regex / Rule-Based Extraction	Meta Nougat Pipeline
Input Type	Extracted plain text strings	Raw page images (Pixels)
Multi-Column Handling	Often mixes left & right columns	Correctly reads structural columns
Section Headings	Fails if title format isn't predicted	Converts headings into Markdown (#, ##)
Math & Tables	Turns equations into broken text	Renders LaTeX equations natively
Error Handling	Brittle; fails silently on edge cases	Smoothly outputs structured Markdown




found many problems within the papers , forma , tables , etc will speak more on it excluded tables are 
able to be traced via excluded tables_log.json




creation of ground trurh (phase 3) to test aganist

What the eval set actually needs

Recall the three buckets from the CRAG design — each question needs a clear expected routing outcome, since that's what you'll check your graph against later:

Should route "Correct" — clearly, fully answerable from your corpus with no ambiguity
Should route "Ambiguous" — partially covered; corpus has something relevant but incomplete, or scattered across sources with gaps
Should route "Incorrect" — genuinely outside your corpus, should trigger full web fallback
Target: ~20-30 questions total, roughly balanced across the three buckets
How to build each bucket concretely, given your actual corpus

Bucket 1 ("Correct") — easiest to write, do this first
Pick 8-10 papers you know are in your corpus (you've been staring at CLIP-Reward, ClipCap, DataComp-related ones already) and write direct factual questions about their core contributions — things you already confirmed retrieve well in your verification tests. E.g. "What reward signal does RLCF use for test-time adaptation?" (you already saw this retrieves cleanly).

Bucket 3 ("Incorrect") — also straightforward, do this second
Two reliable sources for genuinely out-of-corpus questions:

Ask about a paper/technique published after your corpus's cutoff — check your manifest's most recent published dates and pick a topic newer than that
Ask about an adjacent-but-different subfield your corpus deliberately excluded — e.g. if your corpus is VLM/CLIP/captioning specific, ask something from pure NLP-only territory (e.g. a question purely about tokenizer design with zero vision component) that shouldn't have real coverage

Bucket 2 ("Ambiguous") — hardest to write, do this last, after 1 and 3 are locked
This is trickier because it's not "present" or "absent" but "partially present." Two reliable patterns:

A question whose answer spans two different papers' partial contributions — corpus has fragments of the answer but no single chunk fully answers it
A comparison question ("how does X's approach differ from Y's") where X is well-covered but Y is only mentioned in passing (e.g. in someone else's Related Work section) — retrieval will find something but it's clearly incomplete
Format — build this as a structured file, not just a list

Each entry needs, at minimum:

json
{
  "question": "...",
  "expected_routing": "correct" | "ambiguous" | "incorrect",
  "expected_answer_notes": "brief note on what a correct answer should mention",
  "source_papers": ["arxiv_id if applicable"]
}

That expected_answer_notes field matters — when you eventually run the full eval, you're not just checking "did it route correctly," you're also spot-checking "was the final answer actually good," and you want a quick reference for what "good" means without re-reading the paper each time.