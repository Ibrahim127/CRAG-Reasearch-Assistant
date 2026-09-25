# CRAG Research Assistant

<<<<<<< HEAD
If you want to note this in your README's methodology section, it's a perfectly legitimate line: "considered SPECTER2 but used SPECTER for simpler, dependency-light integration via sentence-transformers."



This is a strong, well-evidenced result for your README: "switched from a general-purpose embedding model to a retrieval-tuned one (BGE) after discovering that both general-purpose (MiniLM) and domain-tuned-but-wrong-task (SPECTER) models failed to surface directly relevant, confirmed-present content — the fix was matching the model's training objective (query-passage retrieval) to the actual task, not just chasing domain specificity." That's a genuinely good engineering narrative, not just "I swapped a model."




This is a legitimate, evidence-backed embedding model decision now: 11/10 hits across correct+ambiguous (accounting for correct_03's regression), massive improvement over MiniLM's baseline, and a real "before/after" number for your README (e.g. "expected-paper retrieval rate: X% with MiniLM → Y% with BGE, across a 14-question eval set spanning correct/ambiguous/incorrect routing categories").

Retrieval is now in genuinely solid shape. Want to check the incorrect-bucket results for false positives, or move on to Phase 4 (LangGraph skeleton) now that this foundation is validated?



Phase 4 — LangGraph Skeleton, Broken into Checkpoints

Building the whole graph in one shot means if something's broken, you don't know if it's your state schema, your node logic, your edge routing, or your embeddings all over again. Given how much value the incremental-testing pattern has already caught (table bugs, heading bugs, embedding mismatch), let's keep doing that here.

Checkpoint 1 — State schema + no-op node skeleton, compiles and routes correctly with fake data

Goal: prove the graph structure is wired correctly before any real logic exists.

Define CRAGState (TypedDict)
Every node is a stub that just prints its name and passes state through unchanged
Conditional edge functions use hardcoded/fake routing values you set manually in state (not real grades yet)
Run the compiled graph 3 times with different fake retrieval_action values ("correct", "ambiguous", "incorrect") and confirm it visits the nodes you expect each time
Pass condition: all three paths route correctly with dummy data, no LangGraph plumbing errors
Checkpoint 2 — Real retrieval node, using your actual BGE-backed Chroma setup
Swap the retrieve stub for real Chroma query logic (reuse code from run_eval.py)
Everything downstream still stubbed
Pass condition: run one real question through the graph, confirm state["retrieved_docs"] actually contains real chunks from your corpus
Checkpoint 3 — Real grading node + real routing decision
Implement the per-chunk LLM grading call with structured output
Implement the aggregation function (plain Python, no LLM) that turns grades into correct/ambiguous/incorrect
Pass condition: run 3-4 of your eval questions (one from each bucket) through just retrieve→grade, print the routing decision, compare against expected_routing in your eval set



Checkpoint 4 — Web fallback branch (query rewrite + web search + merge)
Implement rewrite, web search (Tavily/SerpAPI), and merge-context logic
Pass condition: run an incorrect-bucket question through the full ambiguous/incorrect path, confirm 
final_context actually contains web results, not empty/garbage



Checkpoint 5 — Generation node
Real LLM call using final_context
Pass condition: run a correct-bucket question end-to-end, read the generated answer, sanity-check it's actually using the retrieved content (not hallucinating around it)
Checkpoint 6 — Hallucination check + retry loop
Implement the grounding check and retry edge with retry_count cap
Pass condition: deliberately feed a bad/irrelevant context to generate once, confirm the hallucination checker catches it and triggers a retry, and confirm the retry cap actually stops it after 2 attempts (not an infinite loop)
Checkpoint 7 — Full eval run
Run your full eval_set.json through the real compiled graph
Compare actual_routing vs expected_routing per question — this is where the run_full_pipeline() stub from your eval script finally gets used for real

This gives you 7 concrete stopping points, each independently testable, so if something breaks you know exactly which layer to look at. Want to start with Checkpoint 1 — I'll walk you through the state schema and stub-node wiring first?



Worth documenting, not necessarily fixing right now

This is genuinely good material for your README's "limitations and design decisions" section: "grading model choice involves a real precision/recall trade-off — a 3B local model was too lenient (false positives on clearly out-of-scope content), while a 7B model was stricter but occasionally over-cautious on borderline-relevant context chunks. Chose the 7B model since its failure mode (mis-routing correct→ambiguous) degrades gracefully, unlike the 3B model's failure mode (mis-routing incorrect→correct), which risks presenting ungrounded content as trustworthy." That's a substantive, defensible engineering judgment call — exactly the kind of thing that reads well in an interview.



Let's go through all 7, one at a time, with what to actually build and test at each stage.

Checkpoint 1 — Skeleton graph, fake data, prove the wiring works

Goal: confirm LangGraph itself is set up correctly before any real logic touches it.

python
# src/graph/state.py
from typing import TypedDict, Literal, Optional, List

class GradedDoc(TypedDict):
    content: str
    metadata: dict
    grade: Literal["relevant", "irrelevant"]
    reasoning: str

class CRAGState(TypedDict):
    question: str
    rewritten_query: Optional[str]
    retrieved_docs: List[GradedDoc]
    web_docs: List[dict]
    retrieval_action: Optional[Literal["correct", "ambiguous", "incorrect"]]
    final_context: List[dict]
    generation: Optional[str]
    hallucination_status: Optional[Literal["grounded", "not_grounded"]]
    retry_count: int
python
# src/graph/nodes.py -- ALL STUBS for now
def retrieve(state):
    print("[NODE] retrieve")
    return {}  # no-op, don't change state yet

def grade_documents(state):
    print("[NODE] grade_documents")
    return {}

def rewrite_query(state):
    print("[NODE] rewrite_query")
    return {}

def web_search(state):
    print("[NODE] web_search")
    return {}

def merge_context(state):
    print("[NODE] merge_context")
    return {}

def generate(state):
    print("[NODE] generate")
    return {"retry_count": state["retry_count"] + 1}

def check_hallucination(state):
    print("[NODE] check_hallucination")
    return {}
python
# src/graph/edges.py
def route_after_grading(state):
    # TEMPORARY: read a fake value you set manually in the test, not real grades yet
    return state["retrieval_action"]

def route_after_hallucination_check(state):
    if state["hallucination_status"] == "grounded":
        return "end"
    if state["retry_count"] >= 2:
        return "end"
    return "retry"

Wire it exactly as in the full design from last message, then test with 3 manual fake states:

python
# test it
for fake_action in ["correct", "ambiguous", "incorrect"]:
    print(f"\n--- Testing routing: {fake_action} ---")
    result = app.invoke({
        "question": "test",
        "retrieval_action": fake_action,
        "retry_count": 0,
        "hallucination_status": "grounded",  # force immediate end, skip retry loop for now
        "retrieved_docs": [],
        "web_docs": [],
        "final_context": [],
        "rewritten_query": None,
        "generation": None,
    })

Pass condition: "correct" should print retrieve → grade_documents → merge_context → generate → check_hallucination, skipping rewrite_query/web_search. "ambiguous" and "incorrect" should print those two extra steps. If the print order matches this for all three, the graph structure itself is correct — move on.

Checkpoint 2 — Real retrieval node

Replace only retrieve:

python
def retrieve(state):
    query_vec = embed_query(state["question"], model)  # your BGE embedder from run_eval.py
    hits = collection.query(query_embeddings=[query_vec], n_results=5)

    docs = []
    for doc, meta in zip(hits["documents"][0], hits["metadatas"][0]):
        docs.append({"content": doc, "metadata": meta, "grade": None, "reasoning": None})

    return {"retrieved_docs": docs}

Run one real question through the graph (everything else still stubbed, retrieval_action still manually forced). Pass condition: print state["retrieved_docs"] after invoking — confirm real chunk content and metadata from your corpus show up, not empty lists.

Checkpoint 3 — Real grading + real routing

Two pieces: the LLM call, and the pure-Python aggregation.

python
from pydantic import BaseModel
from typing import Literal

class DocGrade(BaseModel):
    relevance: Literal["relevant", "irrelevant"]
    reasoning: str

def grade_documents(state):
    graded = []
    for doc in state["retrieved_docs"]:
        prompt = f"""You are grading whether a retrieved document chunk is relevant to a question.
Question: {state['question']}
Chunk section: {doc['metadata']['section_name']}
Chunk content: {doc['content']}

Judge topical relevance only, not correctness. Is this chunk relevant?"""

        result = small_model.with_structured_output(DocGrade).invoke(prompt)
        doc["grade"] = result.relevance
        doc["reasoning"] = result.reasoning
        graded.append(doc)

    return {"retrieved_docs": graded}
python
def route_after_grading(state):
    grades = [d["grade"] for d in state["retrieved_docs"]]
    if not grades or grades.count("relevant") == 0:
        return "incorrect"
    elif grades.count("relevant") == len(grades):
        return "correct"
    return "ambiguous"

Now remove the manual retrieval_action override from your test invocations — this is real end-to-end routing now.

Pass condition: take one question from each of your eval buckets (correct_01, ambiguous_01, incorrect_01), run retrieve→grade only, print route_after_grading(state) and compare to expected_routing in your eval set. Don't expect 100% match yet — grading prompt quality varies — but confirm it's in the right ballpark (e.g. correct_01 shouldn't come back "incorrect").

Checkpoint 4 — Web fallback branch
python
def rewrite_query(state):
    prompt = f"Rewrite this question into a concise, keyword-focused web search query: {state['question']}"
    rewritten = small_model.invoke(prompt).content
    return {"rewritten_query": rewritten}

def web_search(state):
    from tavily import TavilyClient
    client = TavilyClient(api_key=TAVILY_API_KEY)
    results = client.search(state["rewritten_query"], max_results=5)

    web_docs = [
        {"content": r["content"], "metadata": {"source": r["url"], "title": r["title"]}}
        for r in results["results"]
    ]
    return {"web_docs": web_docs}

Pass condition: run incorrect_01 (FlashAttention-3) through retrieve→grade→rewrite→web_search, print state["web_docs"] — confirm you get real, on-topic web content, not empty results or an API error.

Checkpoint 5 — Merge + Generate
python
def merge_context(state):
    action = state["retrieval_action"]
    if action == "correct":
        ctx = [d for d in state["retrieved_docs"] if d["grade"] == "relevant"]
    elif action == "ambiguous":
        relevant_local = [d for d in state["retrieved_docs"] if d["grade"] == "relevant"]
        ctx = relevant_local + state["web_docs"]
    else:  # incorrect
        ctx = state["web_docs"]
    return {"final_context": ctx}

def generate(state):
    context_text = "\n\n".join(
        f"[{c['metadata'].get('source_paper', c['metadata'].get('source',''))}] {c['content']}"
        for c in state["final_context"]
    )
    prompt = f"""Answer the question using only the context below. Cite sources by paper/title.
Context:
{context_text}

Question: {state['question']}"""

    answer = large_model.invoke(prompt).content
    return {"generation": answer, "retry_count": state["retry_count"] + 1}

Pass condition: run correct_01 fully through retrieve→grade→merge→generate, read the actual generated answer — confirm it correctly cites RLCF's reward mechanism, matches your expected_answer_notes.

Checkpoint 6 — Hallucination check + retry
python
class GroundingCheck(BaseModel):
    grounded: Literal["grounded", "not_grounded"]
    reasoning: str

def check_hallucination(state):
    context_text = "\n\n".join(c["content"] for c in state["final_context"])
    prompt = f"""Does this answer's claims follow from the given context? Context:
{context_text}

Answer: {state['generation']}"""

    result = large_model.with_structured_output(GroundingCheck).invoke(prompt)
    return {"hallucination_status": result.grounded}

route_after_hallucination_check is already written from Checkpoint 1.

Pass condition — deliberately break it first: manually set state["final_context"] to something unrelated (e.g. web_docs from a totally different question) before calling generate, confirm check_hallucination correctly flags "not_grounded" and the graph loops back to generate — then confirm after 2 retries it stops (retry_count >= 2 forces "end"), not an infinite loop.

Checkpoint 7 — Full eval run

Swap the commented-out run_full_pipeline() stub in your run_eval.py for real calls to app.invoke(...), run all 14 questions, and compare actual_routing against expected_routing — this is your first real, automated CRAG accuracy number.














It's a separate LLM call that runs *after* generation, checking whether the answer actually stays within the bounds of the context it was given — not checking truth against the real world, just checking whether the model made things up beyond what it was handed.

## The mechanism, step by step

1. **`generate` produces an answer** using only `final_context` (the merged chunks from retrieval + possibly web search)
2. **`check_hallucination` takes that answer and the same `final_context`**, and asks a fresh LLM call one question: *"does every factual claim in this answer actually appear in this context?"*
3. It returns a structured verdict — `"grounded"` or `"not_grounded"` — plus a short reasoning string explaining why
4. That verdict decides what happens next via `route_after_hallucination_check`:
   - `"grounded"` → done, return the answer
   - `"not_grounded"` and under the retry cap → loop back to `generate` and try again
   - `"not_grounded"` but retries exhausted → give up and return whatever was last generated anyway (better to return something, flagged as unverified, than nothing)

## Why this is a separate call rather than just trusting `generate`

Because an LLM grading its own output in the same breath tends to be overconfident — asking a **fresh call**, with the specific narrow framing of "check this against the context, nothing else," catches cases where `generate` drifted: added a detail it half-remembers from training data rather than the actual retrieved chunk, or stated something the context didn't actually say. It's the same reason your `grade_documents` node is a separate call from `generate` — narrow, single-purpose LLM calls tend to be more reliable than asking one call to both produce and self-critique in one shot.

## The important nuance: "grounded" ≠ "true"
This check only verifies the answer is **faithful to the provided context** — it says nothing about whether the context itself is correct, complete, or even relevant to the question. If `retrieve`/`grade_documents` pulled in the wrong papers, an answer can be perfectly "grounded" in that wrong context and still be a bad answer. This check catches one specific failure mode (the model inventing things beyond its sources) — it's not a general correctness check, and it's worth stating that limitation explicitly in your README.

## Why the retry loop specifically, not just a one-shot check
If the first attempt hallucinates, simply regenerating with the exact same prompt and context often produces the same result again (LLMs are deterministic-ish at `temperature=0`). The retry gives `generate` another shot, and in practice varies slightly due to how conversation state feeds back in — but more importantly, the retry cap (`MAX_RETRIES`) exists specifically so a persistently-hallucinating case doesn't loop forever; it fails safe by returning the last attempt rather than hanging.

Does that answer what you were looking for, or did you want to dig into a specific part — like exactly what the grounding prompt says, or how the retry actually changes generate's behavior on the second pass?












Robust Refusal Property
During evaluation on out-of-context queries (e.g., FlashAttention-3 queries against unrelated context), the generator model (qwen3:8b) strictly adhered to prompt boundaries—explicitly stating that the context lacked sufficient information rather than fabricating an answer. The evaluator correctly scored these honest refusals as grounded, demonstrating strong resistance to hallucination out-of-the-box.

Verification of the Retry Mechanism
Because the model's refusal safety prevented natural hallucination triggers during live runs, the fallback and retry loop (route_after_hallucination_check) was verified using unit tests with synthetic state injections (python test_checkpoint6.py). This verified:

Under-cap retries: Correctly routes to regenerate / retrieve when hallucination == "not_grounded" and retry_count < MAX_RETRIES.

Max-retry capping: Terminates cleanly to END precisely when retry_count >= MAX_RETRIES, preventing infinite control loops.









the hallucination chekcer looping back to the genearte step rather than the retrevakl stage 
because it checkss if the model hallucinated beyond its given context from the retrival stage and makes it retry generating from the context again 

we can implement a loop that goes back to the retrival stage instead as improvment so we can catch bad retriving and ensuring the best results this would tank the performance tho




Neither, really — and that's actually an important nuance to understand before deciding whether to build the extended version.

## The hallucination check doesn't diagnose *why* context was insufficient
`check_hallucination` only sees the final answer and the final context — it has no visibility into what happened upstream. A `"not_grounded"` verdict is a **symptom**, not a diagnosis. It could mean any of several different upstream failures, and the system as designed can't tell which:

1. **Retrieval genuinely missed relevant chunks** — the right content exists in your corpus, but similarity search didn't surface it in top-k (this is exactly the VLRM/BEiTScore problem you debugged earlier — a real retrieval gap)
2. **Retrieval found the right chunks, but the grader wrongly marked them irrelevant** — a grader miscalibration, discarding good context that was actually there
3. **The grader correctly kept relevant chunks, but they're genuinely incomplete** — partial coverage, which is legitimately the `ambiguous` case working as intended, just not fully sufficient
4. **The question is truly unanswerable from your corpus or the web** — not a failure at all, just a hard question

## So what would "loop back to retrieval" actually mean?
If you rewire the retry edge to go back to `rewrite_query`/`web_search` instead of `generate`, you're not correcting a specific identified failure — you're just saying *"whatever happened, try casting a wider net via web search."* It's a blanket fallback, not a targeted fix. It would help case 1 and case 3 (genuinely missing content — a fresh web search might find what local retrieval couldn't), but it wouldn't help case 2 at all (the content was already there, re-searching won't fix a grader that undervalued good context), and it can't help case 4 (nothing to find).

## If you actually want to distinguish these cases
That requires more instrumentation than the runtime graph currently has — logging `retrieval_action`, the individual chunk grades + reasoning, and the hallucination verdict together per question, then manually inspecting patterns (which is exactly the kind of manual diagnostic work you did earlier for the VLRM/BEiTScore misses). That's an offline analysis exercise, not something `check_hallucination` itself can resolve at runtime — an LLM checking "is this grounded" isn't equipped to also answer "whose fault was it that it wasn't."

## My actual recommendation
Given this ambiguity, I'd lean toward **not** building the retry-back-to-retrieval extension right now — it adds complexity without a clear mechanism for knowing whether it'll actually help in a given case, and as discussed, the current refusal-on-insufficient-context behavior is already a safe, honest failure mode. Document this exact limitation in your README instead — it's a substantive, well-reasoned design discussion ("the hallucination retry loop doesn't distinguish retrieval misses, grader miscalibration, or genuinely unanswerable questions, since the checker only observes the final answer, not the upstream pipeline state") — that's a stronger thing to say in an interview than having built an untargeted fallback that might not even help.

Want to move on to **Checkpoint 7** (full eval run) with this documented as a known limitation, or explore the diagnostic-logging idea further first?
=======
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
>>>>>>> daf3df83d60f01970ee6c5b642fefba6062b22cd
