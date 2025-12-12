
# RAG Evaluation & Quality Assurance Pipeline (Python)

## Description
This repository contains a production-grade evaluation pipeline for Retrieval-Augmented Generation (RAG) systems.  
It ingests chat conversations and vectorized context, evaluates each AI response across multiple independent metrics (relevance, completeness, hallucination/factual accuracy, entity & numeric consistency), estimates token usage and cost, flags items for human review, and exports machine-friendly reports (JSON + CSV) for monitoring, dashboards, or CI.

## How it Works — High Level
1. Load chat conversation JSON and context JSON.  
2. Initialize models once (embedding bi-encoder, cross-encoder, spaCy, tokenizer) via a singleton cache to avoid repeated loads.  
3. For each User → AI turn:
   - Retrieve top-K contexts by embedding similarity.
   - Evaluate relevance using a cross-encoder.
   - Check answer-type (format) heuristics.
   - Measure semantic completeness (coverage of query concepts).
   - Detect hallucination via multi-layer grounding checks (semantic, entity, numeric, unsupported claims).
   - Estimate token usage and approximate USD cost.
   - Apply thresholds → flag for human review if needed.
4. Save a detailed JSON (per-turn breakdown) and a CSV summary.

## Detailed Technical Workflow (Deep)

# ![Flow Structure](./gemini_image.png)

### 1) Model Bootstrapping (Singleton Cache)
- **Goal:** Load heavy models once to keep per-turn latency low.
- **Models & Tools:**
  - Embedding Bi-Encoder: `all-MiniLM-L6-v2` — fast encoding for retrieval and semantic comparisons.
  - Cross-Encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2` — precise relevance scoring on (query, response) pairs.
  - NLP: `spaCy en_core_web_sm` — entity recognition, POS tagging, sentence segmentation.
  - Tokenizer: `tiktoken` (e.g., `cl100k_base`) — exact token counts for cost estimation.
  - Utilities: `nltk` (punkt), `numpy`, `pandas`, `scikit-learn`, `tqdm`.

**Why:** avoids repeated disk/GPU loads and stabilizes latency.

### 2) Conversation Parsing
- Expect a JSON file with a `conversation_turns` array; each element has `role`, `message`, optional `turn`.
- The parser pairs each `User` turn with the following `AI/Chatbot` message to form evaluation tasks with `turn_number`, `user_query`, and `ai_response`.

### 3) Context Retrieval (Vector Search)
- Encode query using the bi-encoder.
- Encode all context item texts (or reuse stored vectors).
- Compute cosine similarities and select top-K contexts (default K=3).
- Return both the top context texts and similarity scores.

**Production note:** for very large corpora, replace local encoding with vector DB (FAISS, Milvus, Pinecone) retrieval; retrieval remains the same logic and is fully parallelizable.

### 4) Relevance Scoring
- Use the cross-encoder on `(query, response)` to obtain a logit, apply sigmoid → relevance [0,1].
- Cross-encoder improves precision vs. embedding-only matches because it jointly models pairwise interactions.

### 5) Answer-Type Matching (Heuristic)
- Heuristics detect expected response shape:
  - `how to` → steps, ordering words (step, first, then).
  - `list` → enumeration markers (numbers, bullets, commas).
  - `compare` → contrast words (while, whereas, but).
  - `yes/no` → short confirmation in the first 50 chars.
- If mismatch → flag "Wrong Answer Type".

### 6) Completeness (Semantic Coverage)
- Use spaCy to extract salient query concepts (nouns, verbs, proper nouns) excluding stopwords.
- Embed each concept and response sentences.
- For each concept, find the best-matching response sentence via cosine similarity.
- If match > threshold (e.g., 0.45), consider concept covered.
- Completeness = covered_concepts / total_concepts.

**Robustness:** embedding-based checks tolerate paraphrase and synonyms.

### 7) Hallucination & Factual Accuracy (Multi-Layer)
Compute several signals then aggregate:
1. **Semantic Grounding:** average similarity of response sentences vs. combined retrieved context embedding.
2. **Entity Consistency:** overlap of named entities extracted from response and context (ignore DATE/TIME/CARDINAL noise).
3. **Numeric Consistency:** extract numeric tokens (integers, floats, percentages) and match response numbers to context numbers within tolerance (±5%).
4. **Unsupported Claims:** sentences with very low similarity to context (<0.35) flagged as unsupported.

Aggregate into `factual_accuracy = 0.4*semantic + 0.3*entity + 0.3*numeric`. `hallucination_score = 1 - factual_accuracy`.

### 8) Token & Cost Estimation
- Count tokens: `input_tokens = len(tokenizer.encode(query + context))`, `output_tokens = len(tokenizer.encode(response))`.
- Pricing table (example):
  - `gpt-3.5-turbo`: input $0.0005/1k, output $0.0015/1k
  - `gpt-4o`: input $0.0025/1k, output $0.01/1k
- Estimated USD cost = (input_tokens/1000 * input_price) + (output_tokens/1000 * output_price).

### 9) Flagging & Output
- Default thresholds (configurable):
  - min_relevance = 0.6
  - min_completeness = 0.6
  - max_hallucination = 0.4
  - min_factual = 0.7
- If any threshold violated or answer-type mismatch → `requires_human_review = True` and record `failure_reasons`.
- Save detailed `EvaluationResult` objects to JSON; export summary metrics to CSV for analytics.

## Advantages — Parallelism, Microservice, Scalable, Fast & Cheap
### Parallelism
- Each evaluation turn is independent and can run concurrently:
  - Thread pools, process pools, distributed worker queues (Celery, RabbitMQ, Kafka).
  - Batch embedding computations improve throughput on GPU.

### Microservice Architecture
- Modules are decoupled and can be deployed as services:
  - Retrieval service (vector DB)
  - Relevance scoring service (cross-encoder)
  - Hallucination/factuality service
  - Cost estimation service
  - Export/aggregation service

### Scalability
- Offload vector search to specialized vector DBs; autoscale workers for evaluation tasks.
- Supports batch and streaming evaluation, millions of turns, and multiple clients.

### Fast & Cheap
- No full LLM inference required for evaluations.
- Uses small/efficient models for embeddings and cross-encoder scoring.
- Running costs are mainly model inference for small models — dramatically lower than full LLM calls.

## Libraries & Models (and their use)
- **sentence-transformers / all-MiniLM-L6-v2** — fast embeddings for retrieval and semantic checks.
- **sentence-transformers / cross-encoder/ms-marco-MiniLM-L-6-v2** — high-precision relevance scoring.
- **spaCy (en_core_web_sm)** — entity recognition, POS tagging, sentence segmentation.
- **nltk (punkt)** — sentence tokenization fallback.
- **tiktoken** — token counting for cost estimation.
- **numpy, pandas, scipy, scikit-learn** — numerical ops, CSV export, similarity metrics.
- **tqdm** — progress indicators.

## Why This Approach Is Optimal
- **Fast:** small models and cached initialization yield low latency per turn.
- **Low-Cost:** avoids expensive LLM inference; uses local/efficient models.
- **Modular:** microservice-ready, easy to scale and integrate.
- **Robust:** multi-signal evaluation reduces false alarms and provides actionable diagnostics.
- **Versatile:** switch retrieval backends, adjust thresholds, or add metrics with minimal changes.

## Impacts & Benefits
- Significantly reduces manual QA by surfacing only flagged turns.
- Detects hallucinations and factual errors before release.
- Provides per-turn cost visibility for budgeting experiments.
- Enables model A/B testing and operationalized metrics for production RAG systems.
- Improves trustworthiness and reliability of AI-driven systems.

## Proper Steps to Run the Code (Exact)
1. Create and activate a virtual environment (recommended):
   - macOS / Linux:
     ```
     python -m venv .venv
     source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```

2. Install requirements:
```
pip install --upgrade pip
pip install sentence-transformers spacy nltk scikit-learn tiktoken numpy pandas tqdm scipy
```
3. Download NLP models:
```
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt
```
4. Prepare input files in the working directory (or update paths in the script):
- `sample-chat-conversation-01.json` — must have `conversation_turns` list with `role` and `message`.
- `sample_context_vectors-02.json` — must contain `data.vector_data` or a list of context objects with `text` fields.

5. Run: #python eval_pipeline.py

6. Check outputs:
- `eval_results_refined.json` (detailed per-turn results)
- `eval_results_refined.csv` (summary table)

## Production Tips
- Replace local context encoding with a vector DB for large-scale retrieval.
- Batch-encode queries and contexts to maximize GPU efficiency.
- Expose components as microservices and autoscale based on queue depth.
- Store flagged turns in an annotation platform (Label Studio, internal dashboard) for human review loop and model improvement.

## Thank You
This pipeline is built to be fast, cheap, microservice-friendly.
