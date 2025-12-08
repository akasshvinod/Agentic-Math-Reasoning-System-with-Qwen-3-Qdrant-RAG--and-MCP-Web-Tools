# Agentic Math Reasoning System with Qwen 3, Qdrant RAG, and MCP Web Tools

An industry-grade, human-aligned **Math Tutor Agent** with:

- LangGraph-based orchestration (node/graph pipeline)
- Local RAG over Hendrycks MATH using Qdrant + BGE
- MCP web search tools (Tavily, Wiki, WebFetch)
- Primary reasoning LLM: **Qwen 3 32B** 
- Separate verification LLM
- Human-in-the-loop feedback and logging

---

## 📌 Overview

This project implements an **Agentic Retrieval-Augmented Generation (RAG)** system designed to behave like a university-level mathematics professor.

The system can:

- Understand a wide range of math questions (algebra, calculus, limits, series, word problems)
- Retrieve relevant knowledge from a **local Qdrant vector database** (Hendrycks MATH)
- Fall back to **web search via MCP** (Tavily, Wikipedia, WebFetch) when local context is weak
- Rerank hybrid (local + web) results using a cross-encoder
- Generate **step-by-step mathematical reasoning** with **Qwen 3 32B** as the primary solver LLM
- Validate reasoning using a separate **verification model**
- Accept **Human-in-the-Loop (HITL)** feedback and log it for future training / analysis

The agent is built with **LangGraph (2025)** for node-based orchestration and emphasizes production-grade reliability, safety, and modularity.

---

## 🌟 Key Features

### ✅ 1. Input Guardrails

- Ensures only valid **mathematical** queries enter the pipeline  
- Rejects unsafe / irrelevant / non-math prompts  
- Detects math-related keywords, symbols, and patterns  
- Keeps the downstream reasoning stack focused and safe

### ✅ 2. Query Breaking & Multi-Step Decomposition

- Uses a lightweight LLM (e.g. Llama 3.1 8B instant via Groq) to:
  - Split complex, multi-part problems into atomic sub-questions
  - Normalize expressions to more retrieval-friendly forms
  - Improve recall for both local RAG and MCP web search

### ✅ 3. Local RAG Retrieval (Qdrant + BGE Large v1.5)

- Embeddings: **BAAI/bge-large-en-v1.5** (1024-dim)  
- Vector search over **Hendrycks MATH** stored in Qdrant (running in Docker)
- Optional filters on:
  - `topic`
  - `difficulty`
  - `subject`
- Handles empty / weak results with:
  - Smart fallback logic
  - **RAG quality threshold** (e.g. `RAG_THRESHOLD=0.80`) to decide when to trigger web search

### ✅ 4. MCP Hybrid Web Search (Tavily + Wiki + WebFetch)

When local RAG is weak:

- The graph automatically triggers **MCP search**:
  - `tavily_search` – high-quality web search via Tavily API  
  - `wiki_search` – structured Wikipedia snippets  
  - `web_fetch` – fetches full-page content for the best URL
- Tools are provided by a **FastMCP** server (`math_agent_tools`)
- Called from the agent using `MultiServerMCPClient` (with a robust subprocess bridge)

### ✅ 5. Cross-Encoder Hybrid Reranker

- Reranks both **local RAG** and **web search** results
- Uses a BGE-based reranker model to produce a final ordered list of Documents
- Adaptive logic:
  - If local RAG is strong: fuse local + web results
  - If local RAG is weak: rely primarily on web results

### ✅ 6. Reasoning LLM (Primary Solver – Qwen 3 32B)

- **Primary reasoning engine:** `Qwen 3 32B` (configured in `get_llm` / `run_llm`)  
- Generates:
  - Clear, step-by-step reasoning  
  - Plain-text explanations (no LaTeX required for the client)  
  - Final boxed answer with a short justification
- Can be configured via:
  - Model name
  - Temperature
  - Max tokens
  - Streaming flag

### ✅ 7. Verification LLM (Safety + Math Correctness)

- Separate verification model (e.g. a Groq-hosted model or another OpenRouter model)
- Evaluates:
  - Mathematical correctness
  - Logical consistency
  - Signs of hallucination or missing steps
- Returns structured JSON:

{
  "is_correct": true,
  "issues": [],
  "improved_answer": "..."
}


- The graph uses this JSON to:
  - Decide whether to accept the answer
  - Optionally trigger another retrieval → reasoning loop
  - Increment a `loop_count` with a hard cap

### ✅ 8. Human-in-the-Loop (HITL) Feedback

For every solved query, the system logs:

- `query`
- `final_output` (final answer)
- `retrieval_context` (compact view of top docs)
- `verification` JSON
- `human feedback` (if provided)
- `dspy_eval` (optional DSPy-based evaluation report)

These are appended to:

feedback/logs/feedback_dataset.jsonl

This makes it easy to:

- Inspect model performance
- Fine-tune RAG thresholds
- Train downstream models / DSPy programs

### ✅ 9. Fully Modular LangGraph Pipeline

Core nodes (in execution order):

1. `input_guardrail`
2. `router`
3. `query_breaker`
4. `local_rag`
5. `mcp_search`
6. `hybrid_reranker`
7. `reasoning`
8. `verifier`
9. `output_node`
10. `feedback_node`

Everything runs under a single **LangGraph** compiled graph with `MemorySaver` checkpointing for state persistence per `thread_id`.

---

## 🧱 Project Structure

```
Math_Agent_Project/
│
├── src/
│   ├── graph/
│   │   ├── build_graph.py         # LangGraph pipeline
│   │   ├── state.py               # MathState definition
│   │   └── run_graph.py           # Optional graph runner
│   │
│   ├── nodes/
│   │   ├── input_guardrail.py
│   │   ├── router.py
│   │   ├── query_breaker.py
│   │   ├── local_rag.py
│   │   ├── mcp_search.py
│   │   ├── hybrid_reranker.py
│   │   ├── reasoning.py
│   │   ├── verifier.py
│   │   ├── output_node.py
│   │   └── feedback_node.py
│   │
│   ├── tools/
│   │   ├── qdrant_tool.py         # Qdrant client wrapper
│   │   ├── local_rag_tool.py      # LangChain Tool for local RAG
│   │   ├── search_mcp_server.py   # FastMCP server (tavily, wiki, web_fetch)
│   │   └── mcp_clients.py         # MultiServerMCPClient manager
│   │
│   ├── embedder/
│   │   └── embedder.py            # BGE embedding loader
│   │
│   ├── hitl/
│   │   ├── feedback_store.py      # JSONL feedback logger
│   │   └── dspy_evaluator.py      # DSPy-based evaluator
│   │
│   ├── tests/
│   │   ├── test_local_rag_tool.py
│   │   ├── test_mcp_search_node.py
│   │   ├── test_reranker.py
│   │   └── test_full_graph.py
│   │
│   ├── run_agent.py               # CLI math tutor interface
│   ├── config.py
│   └── utils/logging.py
│
├── data/
│   ├── raw/        # Original Hendrycks MATH JSONL
│   └── cleaned/    # Normalized / filtered data
│
├── vectorstore/
│   └── qdrant/     # Qdrant storage (Docker volume)
│
├── feedback/
│   └── logs/feedback_dataset.jsonl
│
├── README.md
├── .env
└── requirements.txt

---

## ⚙️ Installation & Setup

### 1. Start Qdrant via Docker

```
docker run -d --name qdrant -p 6333:6333 \
  -v /absolute/path/to/vectorstore/qdrant:/qdrant/storage \
  qdrant/qdrant
```

Verify:

curl http://localhost:6333/collections
# → should list hendrycks_maths / hendrycks_math
```

If it already exists after reboot:

```
docker start qdrant

### 2. Create Python Environment

```
conda create -n math-agent python=3.10 -y
conda activate math-agent

pip install -r requirements.txt
```

### 3. Configure `.env`

Example:

# Embeddings / Qdrant
EMBED_MODEL=BAAI/bge-large-en-v1.5

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=hendrycks_maths
RAG_THRESHOLD=0.80

# Primary reasoning LLM: Qwen 3 32B via OpenRouter
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=your_openrouter_key_here
OPENAI_MODEL=qwen/qwen3-32b  # primary reasoning LLM

# Verification LLM (example, Groq or other)
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=openai/gpt-oss-20b

# MCP / Tavily / Wiki
TAVILY_API_KEY=tvly-xxxx
WIKI_API_KEY=xxxx   # if required

Qwen 3 32B is wired as the **primary reasoning LLM** in your `get_llm` / `run_llm` helper and in `reasoning_node`.

---

## 🎯 Running the Agent

### CLI Math Tutor

Use the CLI interface:

python -m src.run_agent

Sample:

```
┌───────────────────────────────────────────────┐
│           Math Agent - CLI Interface         │
└───────────────────────────────────────────────┘
Type your math question (calculus, algebra,
limits, series, etc.).

Commands:
  /exit, /quit   → end session
  /clear         → clear screen

[session id: cli-session-ab12cd34]

You  > lim_{x→0} (tan x – sin x) / x^3

Agent> Thinking...

╭─ ANSWER ────────────────────────────────────────────────╮
│ Summary:
│   The limit lim_{x→0} (tan x – sin x) / x^3 equals 1/2.
│
│ Steps:
│   -  Expand tan x and sin x near 0 using Taylor series.
│   -  Subtract to obtain a leading x^3/2 term plus higher orders.
│   -  Divide by x^3 and take the limit as x → 0 to get 1/2.
╰──────────────────────────────────────────────────────────╯
```

---

## 📊 Example I/O

**Input**

```
Solve for x: x^2 = 16
```

**Output (Qwen 3 32B reasoning)**

```
Summary:
  The equation x^2 = 16 has two real solutions: x = 4 and x = -4.

Steps:
  -  Take square roots on both sides, giving |x| = 4.
  -  Therefore x can be 4 or -4.
```

**Input**

```
Differentiate x^3
```

**Output**

```
Summary:
  The derivative of x^3 with respect to x is 3x^2.

Steps:
  -  Apply the power rule d/dx (x^n) = n·x^(n-1).
  -  Here n = 3, so d/dx (x^3) = 3·x^(3-1) = 3x^2.
```

---

## 🧠 Why This Project Is Industry-Grade

This architecture mirrors modern **agentic AI stacks** used in real-world systems:

- Safety guardrails at the input edge
- Multi-step query decomposition
- Local RAG + web (MCP) hybrid retrieval
- Cross-encoder reranking
- Distinct **reasoning LLM** (Qwen 3 32B) and **verification LLM**
- Human-in-the-loop logging for continual learning
- LangGraph-based modular pipeline with clear, testable nodes


---

## 📈 Future Enhancements

- Train a task-specific reranker on MATH-style problems  
- Integrate a symbolic math engine (e.g. SymPy) as an additional tool  
- Use DSPy to train programs on the feedback logs  
- Add a Streamlit / React frontend for web-based tutoring  
- Wrap the graph in a FastAPI / gRPC service for production deployment

---

## 💬 Contact / Support

If you want help:

- Hardening this for production (API, Docker, CI/CD)
- Optimizing prompts, retrieval, or verification logic
- Positioning this project in your resume / portfolio

you can extend this README or reach out with specific questions about any module or flow.