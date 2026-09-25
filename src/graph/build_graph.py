
from langgraph.graph import StateGraph, END

from src.graph.state import CRAGState
from src.graph.nodes import (
    retrieve,
    grade_documents,
    rewrite_query,
    web_search,
    merge_context,
    generate,
    check_hallucination,
)
from src.graph.edges import route_after_grading, route_after_hallucination_check


def build_graph():
    graph = StateGraph(CRAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("web_search", web_search)
    graph.add_node("merge_context", merge_context)
    graph.add_node("generate", generate)
    graph.add_node("check_hallucination", check_hallucination)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "correct": "merge_context",
            "ambiguous": "rewrite_query",
            "incorrect": "rewrite_query",
        },
    )

    graph.add_edge("rewrite_query", "web_search")
    graph.add_edge("web_search", "merge_context")
    graph.add_edge("merge_context", "generate")
    graph.add_edge("generate", "check_hallucination")

    graph.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination_check,
        {"retry": "generate", "end": END},
    )

    return graph.compile()


app = build_graph()