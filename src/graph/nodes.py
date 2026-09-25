from src.graph.state import CRAGState
from src.graph import config


def _get_groq_model(model_name: str, temperature: float = 0):
    import os
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get(config.GROQ_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"{config.GROQ_API_KEY_ENV} not set -- check your .env file")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=config.GROQ_BASE_URL,
        temperature=temperature,
        timeout=30,
        max_retries=1,  # outer tenacity wrapper handles backoff, keep this low
    )


def _groq_retry():
    """Shared tenacity retry decorator for rate limits / transient server errors."""
    import openai
    from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt

    return retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIStatusError, openai.APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=config.GROQ_RETRY_MIN_WAIT, max=config.GROQ_RETRY_MAX_WAIT),
        stop=stop_after_attempt(config.GROQ_MAX_RETRY_ATTEMPTS),
        reraise=True,
    )


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return "".join(parts)
    return str(content)

_embedder = None
_collection = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print(f"[*] Loading embedding model: {config.EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


def _get_collection(collection_name: str = None):
    """
    Connects to a Chroma collection. Pass collection_name to query a
    specific corpus (e.g. one a user just ingested via
    ingest_documents.py) -- defaults to config.COLLECTION_NAME if not
    given, so existing scripts/tests keep working unchanged.

    Cached per collection name, so switching between corpora within
    the same process (e.g. in the Streamlit app) doesn't require
    reconnecting every call, but also doesn't stick you with the first
    collection you ever queried.
    """
    global _collection_cache
    if "_collection_cache" not in globals():
        _collection_cache = {}

    name = collection_name or config.COLLECTION_NAME

    if name not in _collection_cache:
        import chromadb
        client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        _collection_cache[name] = client.get_collection(name)
        print(f"[*] Connected to Chroma collection '{name}' "
              f"({_collection_cache[name].count()} chunks)")

    return _collection_cache[name]


def retrieve(state: CRAGState) -> dict:
    print("[NODE] retrieve")

    model = _get_embedder()
    collection = _get_collection(state.get("collection_name"))

    prefixed_question = config.QUERY_PREFIX + state["question"]
    query_vec = model.encode(prefixed_question, normalize_embeddings=True).tolist()

    hits = collection.query(query_embeddings=[query_vec], n_results=config.TOP_K)

    docs = []
    for doc, meta in zip(hits["documents"][0], hits["metadatas"][0]):
        docs.append({
            "content": doc,
            "metadata": meta,
            "grade": None,
            "reasoning": None,
        })

    return {"retrieved_docs": docs}


def grade_documents(state: CRAGState) -> dict:
    print("[NODE] grade_documents")

    from pydantic import BaseModel, Field
    from typing import Literal, List
    import openai

    class SingleGrade(BaseModel):
        chunk_number: int = Field(description="The number of the chunk being graded (starting at 1)")
        relevance: Literal["relevant", "irrelevant"] = Field(description="Topical relevance of this chunk")
        reasoning: str = Field(description="Reasoning for the relevance grade")

    class BatchGrade(BaseModel):
        grades: List[SingleGrade] = Field(description="List of grades for each individual retrieved chunk")
        fully_covers_question: bool = Field(description="True if the chunks together cover every part of the user question")
        coverage_reasoning: str = Field(description="Reasoning regarding full context coverage")

    global _grading_model
    if "_grading_model" not in globals() or _grading_model is None:
        _grading_model = _get_groq_model(config.SMALL_MODEL)

    structured_model = _grading_model.with_structured_output(BatchGrade)

    @_groq_retry()
    def _graded_invoke(prompt: str):
        return structured_model.invoke(prompt)

    docs = state["retrieved_docs"]

    chunks_block = "\n\n".join(
        f"--- Chunk {i+1} (section: {doc['metadata'].get('section_name', 'unknown')}) ---\n"
        f"{doc['content']}"
        for i, doc in enumerate(docs)
    )

    prompt = f"""You are grading whether retrieved document chunks are relevant to a question.
You are NOT answering the question -- only judging topical relevance of each chunk.

PER-CHUNK GRADING:
Mark a chunk "relevant" if it discusses the same method, system, or topic the
question is about -- even if it only provides background, setup, or partial
detail rather than the complete answer. Err toward "relevant" when the chunk
is clearly about the right subject matter, even if it doesn't use the exact
wording of the question.
Mark a chunk "irrelevant" only if it is about a genuinely different topic,
method, or paper than what the question is asking about.

SET-LEVEL COVERAGE CHECK (separate from per-chunk grading):
After grading each chunk individually, judge whether the chunks TOGETHER
cover every part of the question. For a simple single-topic question, this
is normally true as long as some chunks are relevant. But for a comparison
or multi-part question (e.g. "how does X compare to Y", "what's the
difference between A and B"), set fully_covers_question=False if the chunks
only discuss ONE side of the comparison and never address the other side --
even if every individual chunk is correctly graded "relevant" to its own
side of the topic.

Question: {state['question']}

{chunks_block}

Grade EVERY chunk above (there are {len(docs)} chunks, numbered 1 to {len(docs)}),
then give your overall fully_covers_question judgment."""

    try:
        result = _graded_invoke(prompt)
    except (openai.RateLimitError, openai.APIStatusError, openai.APIConnectionError) as e:
        print(f"    [!] Groq call failed after retries: {e} -- defaulting all chunks to irrelevant "
              "(fail-safe, does not crash the graph)")
        graded = []
        for doc in docs:
            doc["grade"] = "irrelevant"
            doc["reasoning"] = "Grading failed after repeated retries."
            graded.append(doc)
        return {"retrieved_docs": graded, "retrieval_action": "incorrect"}

    grades_by_number = {g.chunk_number: g for g in result.grades}
    graded = []
    for i, doc in enumerate(docs):
        g = grades_by_number.get(i + 1)
        if g is None:
            print(f"    [!] no grade returned for chunk {i+1}, defaulting to irrelevant")
            doc["grade"] = "irrelevant"
            doc["reasoning"] = "No grade returned by batch call."
        else:
            doc["grade"] = g.relevance
            doc["reasoning"] = g.reasoning
        graded.append(doc)
        print(f"    graded '{doc['metadata'].get('section_name')}' -> {doc['grade']}")

    print(f"    coverage check: fully_covers_question={result.fully_covers_question} "
          f"({result.coverage_reasoning[:100]}...)")

    grade_values = [d["grade"] for d in graded]
    
    relevant_ratio = 0.0
    
    if not grade_values:
        action = "incorrect"
    else:
        relevant_ratio = grade_values.count("relevant") / len(grade_values)
        if relevant_ratio == 0:
            action = "incorrect"
        elif relevant_ratio >= config.CORRECT_THRESHOLD:
            action = "correct"
        else:
            action = "ambiguous"

        if action == "correct" and not result.fully_covers_question:
            print(f"    coverage override: all chunks relevant, but set doesn't fully "
                  f"cover the question -- downgrading correct -> ambiguous")
            action = "ambiguous"

    print(f"    routing decision: {grade_values.count('relevant')}/{len(grade_values)} "
          f"relevant ({relevant_ratio:.0%}) -> {action}")

    return {"retrieved_docs": graded, "retrieval_action": action}


def rewrite_query(state: CRAGState) -> dict:
    print("[NODE] rewrite_query")

    import openai

    global _rewrite_model
    if "_rewrite_model" not in globals() or _rewrite_model is None:
        _rewrite_model = _get_groq_model(config.SMALL_MODEL)

    @_groq_retry()
    def _rewrite_invoke(prompt: str):
        return _rewrite_model.invoke(prompt)

    prompt = f"""Rewrite the following question into a short, keyword-focused
web search query. Remove filler words, keep technical terms and names exactly
as written. Output ONLY the rewritten query, nothing else.

Question: {state['question']}"""

    try:
        result = _rewrite_invoke(prompt)
        rewritten = _extract_text(result.content)
    except (openai.RateLimitError, openai.APIStatusError, openai.APIConnectionError) as e:
        print(f"    [!] Groq call failed after retries: {e} -- falling back to original question as query")
        rewritten = state["question"]

    print(f"    rewritten query: {rewritten!r}")
    return {"rewritten_query": rewritten}


def web_search(state: CRAGState) -> dict:
    print("[NODE] web_search")

    import os
    from tavily import TavilyClient

    global _tavily_client
    if "_tavily_client" not in globals() or _tavily_client is None:
        api_key = os.environ.get(config.TAVILY_API_KEY_ENV)
        if not api_key:
            raise ValueError(
                f"{config.TAVILY_API_KEY_ENV} not set -- check your .env file"
            )
        _tavily_client = TavilyClient(api_key=api_key)

    query = state["rewritten_query"] or state["question"]
    response = _tavily_client.search(query, max_results=config.WEB_SEARCH_MAX_RESULTS)

    web_docs = []
    for r in response.get("results", []):
        web_docs.append({
            "content": r.get("content", ""),
            "metadata": {
                "source_paper": None,
                "section_name": r.get("title", "Web result"),
                "content_type": "web",
                "url": r.get("url"),
            },
        })
        print(f"    web result: {r.get('title')} ({r.get('url')})")

    return {"web_docs": web_docs}


def merge_context(state: CRAGState) -> dict:
    print("[NODE] merge_context")

    action = state["retrieval_action"]

    if action == "correct":
        ctx = [d for d in state["retrieved_docs"] if d["grade"] == "relevant"]
    elif action == "ambiguous":
        relevant_local = [d for d in state["retrieved_docs"] if d["grade"] == "relevant"]
        ctx = relevant_local + state["web_docs"]
    else:  # incorrect
        ctx = state["web_docs"]

    print(f"    merged context: {len(ctx)} items (action={action})")
    return {"final_context": ctx}


def generate(state: CRAGState) -> dict:
    print("[NODE] generate")

    import openai

    global _generation_model
    if "_generation_model" not in globals() or _generation_model is None:
        _generation_model = _get_groq_model(config.LARGE_MODEL)

    @_groq_retry()
    def _generate_invoke(prompt: str):
        return _generation_model.invoke(prompt)

    context_blocks = []
    unique_sources = []
    
    for c in state["final_context"]:
        meta = c["metadata"]
        source_name = meta.get("source_paper") or meta.get("url") or meta.get("section_name") or "Unknown Source"
        
        if source_name not in unique_sources:
            unique_sources.append(source_name)
            
        source_idx = unique_sources.index(source_name) + 1
        
        context_blocks.append(
            f"--- Document [{source_idx}] ---\n"
            f"Source: {source_name}\n"
            f"Content:\n{c['content']}"
        )
        
    context_text = "\n\n".join(context_blocks)

    prompt = f"""You are an expert research assistant. Your task is to provide a comprehensive, cohesive, and well-structured answer to the user's question based strictly on the provided context documents.

INSTRUCTIONS:
1. Synthesize the information into a flowing, cohesive response. Use markdown formatting (headings, bullet points, bold text) to make it highly readable.
2. DO NOT clutter the text with long inline source names or URLs. 
3. Instead, use footnote-style bracketed numbers inline (e.g., [1], [2]) immediately after you state a fact derived from a specific document.
4. At the very end of your response, you MUST include a "### Sources" section that cleanly lists the references corresponding to the numbers used.
5. If the context does not contain enough information to fully answer the question, state what is known and explicitly acknowledge the gaps. Do not hallucinate.

CONTEXT DOCUMENTS:
{context_text}

USER QUESTION: 
{state['question']}

EXPERT ANSWER:"""

    try:
        result = _generate_invoke(prompt)
        answer = _extract_text(result.content).strip()
    except (openai.RateLimitError, openai.APIStatusError, openai.APIConnectionError) as e:
        print(f"    [!] Groq call failed after retries: {e} -- returning fail-safe message")
        answer = "Unable to generate an answer due to repeated API errors."

    print(f"    generated answer ({len(answer)} chars)")
    return {"generation": answer, "retry_count": state["retry_count"] + 1}


def check_hallucination(state: CRAGState) -> dict:
    print("[NODE] check_hallucination")

    from pydantic import BaseModel
    from typing import Literal
    import openai

    class GroundingCheck(BaseModel):
        grounded: Literal["grounded", "not_grounded"]
        reasoning: str

    global _hallucination_model
    if "_hallucination_model" not in globals() or _hallucination_model is None:
        _hallucination_model = _get_groq_model(config.LARGE_MODEL)

    structured_model = _hallucination_model.with_structured_output(GroundingCheck)

    @_groq_retry()
    def _check_invoke(prompt: str):
        return structured_model.invoke(prompt)

    context_text = "\n\n".join(c["content"] for c in state["final_context"])

    prompt = f"""You are checking whether an answer is grounded in the given context.
An answer is "grounded" if every factual claim it makes is directly supported by
the context. An answer is "not_grounded" if it makes claims not present in the
context, contradicts the context, or is unsupported speculation.

Context:
{context_text}

Answer to check:
{state['generation']}

Judge whether this answer is grounded in the context."""

    try:
        result = _check_invoke(prompt)
        status = result.grounded
        reasoning = result.reasoning
    except (openai.RateLimitError, openai.APIStatusError, openai.APIConnectionError) as e:
        print(f"    [!] Groq call failed after retries: {e} -- defaulting to 'grounded' "
              "(fail-safe, avoids an infinite retry loop on an API issue)")
        status, reasoning = "grounded", "Hallucination check skipped due to repeated API errors."

    print(f"    grounding verdict: {status}  ({reasoning[:100]}...)")
    return {"hallucination_status": status}