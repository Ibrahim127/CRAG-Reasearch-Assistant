from src.graph.state import CRAGState


def route_after_grading(state: CRAGState) -> str:
    
    action = state.get("retrieval_action")
    if action not in ("correct", "ambiguous", "incorrect"):
        raise ValueError(
            f"retrieval_action must be 'correct', 'ambiguous', or 'incorrect', got: {action!r}"
        )
    return action


def route_after_hallucination_check(state: CRAGState) -> str:
    from src.graph import config

    if state.get("hallucination_status") == "grounded":
        return "end"
    if state.get("retry_count", 0) >= config.MAX_RETRIES:
        print(f"    retry cap ({config.MAX_RETRIES}) reached -- ending despite not_grounded")
        return "end"
    print(f"    not grounded, retry_count={state.get('retry_count', 0)} -- retrying generate")
    return "retry"