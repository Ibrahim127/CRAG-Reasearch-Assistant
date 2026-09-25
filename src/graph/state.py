from typing import TypedDict, Literal, Optional, List


class GradedDoc(TypedDict):
    content: str
    metadata: dict          # source_paper, section_name, content_type, chunk_index
    grade: Optional[Literal["relevant", "irrelevant"]]
    reasoning: Optional[str]


class CRAGState(TypedDict):
    question: str
    collection_name: Optional[str]  # which Chroma collection to query; defaults to config.COLLECTION_NAME if not set
    rewritten_query: Optional[str]
    retrieved_docs: List[GradedDoc]
    web_docs: List[dict]
    retrieval_action: Optional[Literal["correct", "ambiguous", "incorrect"]]
    final_context: List[dict]
    generation: Optional[str]
    hallucination_status: Optional[Literal["grounded", "not_grounded"]]
    retry_count: int