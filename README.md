# Real-Time RAG Evaluation Pipeline (Lite)

A streamlined, lightweight Python pipeline for evaluating Retrieval-Augmented Generation (RAG) systems in real-time. This tool provides automated, granular scoring for **Relevance**, **Completeness**, **Factual Accuracy (Hallucination)**, **Latency**, and **Cost** without relying on expensive external APIs like GPT-4 for judging.

---

## 🚀 Overview

This pipeline is designed to be embedded within a RAG application. For each user query and AI response, it automatically calculates a set of quality metrics. It is built to be:

* **Fast & Local:** Runs entirely on your machine (CPU/GPU) with no API latency or costs.
* **Accurate without LLMs:** Uses specialized, smaller NLP models (Cross-Encoders, spaCy) instead of using a large LLM as a judge.
* **Explainable:** Provides clear reasons for scores, moving beyond simple "black box" metrics.

## ⚙️ Core Components & Methodology

The pipeline evaluates a `(Query, Response, Retrieved_Context)` triplet across three main pillars.

### 1. Relevance & Completeness

**Goal:** Determine if the response directly answers the user's query and covers all key concepts.

#### **Relevance Score (Cross-Encoder)**
* **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **How it works:** This model is trained specifically for Question-Answering tasks. It takes the Query and Response as a **single pair** of input text. The model "reads" them together, allowing its attention mechanism to understand the deep semantic relationship between the question and the answer.
* **Output:** A raw logit score which is squashed via a Sigmoid function to a 0.0 - 1.0 probability representing relevance.

#### **Completeness Score (Semantic Concept Matching)**
* **Models:** `spaCy (en_core_web_sm)` for NLP, `sentence-transformers/all-MiniLM-L6-v2` for embedding.
* **How it works:**
    1.  **Extract:** spaCy extracts "key concepts" (noun chunks and verbs) from the User Query, ignoring filler words.
    2.  **Embed:** The Bi-Encoder creates a vector for each query concept and for each sentence in the AI Response.
    3.  **Match:** For every query concept, we search for a semantic match (cosine similarity > 0.45) among the response sentences.
* **Output:** The percentage of query concepts that have a semantic match in the response.

### 2. Factual Accuracy & Hallucination

**Goal:** Verify that the AI's response is supported by the retrieved context and contains no fabricated information.

* **Models:** `spaCy (en_core_web_sm)` and `sentence-transformers/all-MiniLM-L6-v2`.
* **How it works:** This score is a weighted average of two checks.
    1.  **Semantic Grounding (50%):** We embed the entire response and the entire retrieved context and calculate their cosine similarity. A low score indicates the response is off-topic compared to the source material.
    2.  **Entity Consistency (50%):** spaCy's Named Entity Recognition (NER) extracts specific entities (persons, organizations, dates, etc.) from both texts. We calculate the percentage of entities in the response that also appear in the context.
* **Output:**
    * **Factual Score:** `(0.5 * Grounding_Score) + (0.5 * Entity_Score)`
    * **Hallucination Score:** `1.0 - Factual_Score`

### 3. Operational Metrics (Latency & Cost)

**Goal:** Track the efficiency and cost of the RAG system.

* **Latency:** Measured as the time taken to process a single turn.
* **Cost:** Uses `tiktoken` (OpenAI's tokenizer) to count the exact tokens in the input (Query + Context) and output (Response). A fixed rate (e.g., $0.0005 per 1k tokens) is applied to estimate dollar cost.

---

## 💡 Why This Approach? (Comparison)

We chose this specific architecture over other common evaluation methods for several key reasons.

### vs. LLM-as-a-Judge (LAAJ)
Using a powerful model like GPT-4 to grade responses is a popular "gold standard" but has significant downsides for real-time production systems.

| Feature | **Our Pipeline** | **LLM-as-a-Judge (GPT-4)** |
| :--- | :--- | :--- |
| **Cost** | **Free** (Runs locally) | **High** (Per-token API costs) |
| **Latency** | **Fast** (Milliseconds) | **Slow** (Seconds per call) |
| **Privacy** | **Secure** (Data stays local) | **Risk** (Data sent to 3rd party) |
| **Reliability** | **Deterministic** (Same input = same score) | **Stochastic** (Scores can vary) |

**Conclusion:** Our approach provides a much faster, cheaper, and more private solution for continuous, real-time monitoring.

### vs. Traditional Metrics (ROUGE, BLEU)
These metrics were designed for translation and summarization tasks where there is a single "correct" reference answer. They are ill-suited for open-ended RAG evaluation.

* **ROUGE/BLEU rely on exact word overlap (n-gram matching).**
* **Problem:** In RAG, a good answer can use completely different words than the context or a reference answer. A response like "You must fix the leak" would get a near-zero score against a reference like "The user needs to repair the dripping faucet," even though the meaning is identical.
* **Our Solution:** We use **Semantic Embedding** models which understand that "fix" and "repair" are synonyms, resulting in accurate scores that reflect meaning, not just word choice.

---

## 🧠 Why These Specific Models?

* **`cross-encoder/ms-marco-MiniLM-L-6-v2`:**
    * **Why?** It is a highly-ranked, compact model specifically fine-tuned on the MS MARCO passage ranking dataset. It is purpose-built for determining the relevance of a passage to a query, making it ideal for our **Relevance** score. It is much more accurate for this task than a standard bi-encoder.
* **`sentence-transformers/all-MiniLM-L6-v2`:**
    * **Why?** This is an excellent, lightweight, all-purpose **Bi-Encoder**. It's incredibly fast and produces high-quality sentence embeddings, making it perfect for our real-time **Semantic Search** (Completeness) and **Grounding** (Hallucination) tasks.
* **`spaCy (en_core_web_sm)`:**
    * **Why?** spaCy is an industrial-standard library for NLP. Its small English model (`en_core_web_sm`) is very fast and provides robust **Named Entity Recognition** and **Noun Chunking**, which are essential for our granular Completeness and Entity Consistency checks.

---

## 📦 Installation & Setup

1.  **Install Python dependencies:**
    ```bash
    pip install sentence-transformers spacy tiktoken numpy pandas tqdm
    ```

2.  **Download the spaCy language model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

## 🏃‍♂️ Usage

The core logic is encapsulated in the `simple_eval.py` script. You can run it directly to evaluate sample JSON data.

1.  Ensure you have your chat logs (`sample-chat-conversation-01.json`) and context data (`sample_context_vectors-01.json`) in the same directory.
2.  Run the script:
    ```bash
    python simple_eval.py
    ```
3.  The script will print a summary of average scores to the console and save the detailed, turn-by-turn results to a CSV file named `simple_eval_results.csv`.

To integrate this into your own application, you can import the `RAGEvaluator` class and call its `evaluate_conversation` method with your own lists of chat turns and contexts.
