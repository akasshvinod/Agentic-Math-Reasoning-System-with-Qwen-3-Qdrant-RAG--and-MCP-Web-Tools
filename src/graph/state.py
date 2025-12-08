from typing import List, TypedDict, Optional, Dict, Any
from langchain_core.documents import Document


class MathState(TypedDict, total=False):
    """
    Shared state passed between nodes in the LangGraph pipeline.

    Compatible with LangGraph MemorySaver via the `memory` field.

    Fields:
        query:               Incoming user question
        is_safe:             Input guardrail result
        is_math:             Router classification
        broken_queries:      QueryBreaker output
        local_results:       List[Document] from Qdrant
        web_results:         Raw MCP search results
        reranked_results:    Hybrid ranked documents
        reasoning_output:    Output of the reasoning LLM
        verification:        JSON string or dict from verifier node
        final_output:        Clean user-facing answer
        needs_web_fallback:  True if local RAG judged weak
        loop_count:          Verification / retry counter
        feedback:            Human-in-the-loop feedback text
        memory:              Checkpoint data stored by MemorySaver
    """

    # USER INPUT
    query: str

    # GUARDRAILS
    is_safe: bool
    is_math: bool

    # QUERY PROCESSING
    broken_queries: List[str]

    # RETRIEVAL
    local_results: List[Document]
    web_results: List[Dict[str, Any]]
    reranked_results: List[Document]

    # GENERATION
    reasoning_output: str
    verification: Any           # JSON string or dict
    final_output: str

    # ROUTING / CONTROL
    needs_web_fallback: bool
    loop_count: int

    # HUMAN-IN-THE-LOOP
    feedback: Optional[str]

    # MEMORY CHECKPOINT
    memory: Optional[Dict[str, Any]]
