# Multi-Agent Orchestration with RAG & NVIDIA Builder Models

An enterprise-grade financial document ingestion, metadata enrichment, and retrieval pipeline powered by **NVIDIA AI Endpoints**, **Qdrant Vector Database**, and **PyTorch RNNs**. 

This system acts as a secure, intelligent backend that processes complex regulatory and financial documents, indexes them semantically, and orchestrates user queries through a rigorous pipeline of security guardrails, intent classification, and query rewriting before performing a vector search.

---

## 🤖 Models Used & Their Purposes

This project heavily leverages NVIDIA AI endpoints and HuggingFace models for various discrete tasks, following a multi-agent architectural pattern:

### 1. `meta/llama-3.1-8b-instruct` (via NVIDIA API)
The primary workhorse LLM used for structured data extraction and logical reasoning across multiple modules:
*   **Document Pipeline (`document_pipeline.py`)**: Powers the `ChatNVIDIA.abatch()` calls to extract high-level document metadata (title, summary, domain) and granular chunk-level metadata (entities, synthetic QA pairs, compliance mandates) using strict Pydantic schemas.
*   **Query Enrichment (`query_enrichment.py`)**: Performs semantic expansion and context-aware query rewriting (based on chat history) to optimize the query for vector retrieval.
*   **Query NLP (`QueryNLP/entity_extraction.py`, `QueryNLP/spelling_correction.py`)**: Extracts named entities from raw queries and performs intelligent spelling correction.
*   **Security Layer**: Drives the logic for the `AccessAuthorizationGuard` (RBAC checking), `PromptInjectionGuard` (adversarial detection), and acts as the Privacy Sanitizer in the `SecurityOrchestrator` to rewrite queries that contain sensitive PII.

### 2. `nvidia/nv-embedqa-e5-v5` (via NVIDIA API)
*   **Semantic Chunking (`document_pipeline.py`)**: Used by the `SemanticChunker` to calculate semantic similarity between sentences, determining the optimal breakpoints to split documents contextually rather than arbitrarily.
*   **Vector Database Indexing (`qdrant_indexer.py`)**: Embeds the enriched document chunks into high-dimensional dense vectors to be stored in the persistent Qdrant database.

### 3. `nvidia/llama-nemotron-embed-1b-v2` (via NVIDIA API)
*   **Query Embedding (`query_embedding.py`)**: A lightweight, highly efficient embedding model used specifically to vectorize incoming user queries rapidly for semantic matching against the Qdrant index.

### 4. `nvidia/llama-3.1-nemoguard-8b-content-safety` (via NVIDIA API)
*   **Content Safety Guard (`SecurityLayer/content_safety_guard.py`)**: A specialized NeMo Guardrails model used exclusively to detect and block toxic, illegal, or unethical content in user prompts.

### 5. `distilbert-base-uncased` + Custom PyTorch BiLSTM
*   **Intent Classification (`QueryNLP/intent_detection.py`, `QueryNLP/query_classification.py`)**: A locally trained PyTorch model. The transformer weights are frozen, and a Bidirectional LSTM is trained on top of it using the synthetic QA pairs generated during document ingestion. It classifies user queries into specific domain intents (e.g., *Regulatory Guideline*, *Annual Report*).

---

## 📦 Module Usage & Architecture

### 1. Document Processing Pipeline (`document_pipeline.py`)
*   **Usage**: Run `python3 document_pipeline.py`
*   **Purpose**: Scans the `Data/` folder for PDFs and TXT files. It loads the text, determines semantic chunk boundaries, extracts structured metadata for every chunk, and saves the highly enriched artifacts to `processed_documents.json` and `processed_documents.jsonl`.
*   **Note**: Features a custom asynchronous sliding-window rate limiter to guarantee API calls stay strictly under 40 RPM.

### 2. Vector Database Indexing (`qdrant_indexer.py`)
*   **Usage**: Run `python3 qdrant_indexer.py`
*   **Purpose**: Reads the enriched `processed_documents.jsonl` artifacts, generates embeddings for them using `nvidia/nv-embedqa-e5-v5`, and bulk-inserts them into a local persistent Qdrant Vector Database collection.

### 3. Security Layer (`SecurityLayer/`)
*   **Usage**: Automatically invoked during the query pipeline via `SecurityOrchestrator`.
*   **Purpose**: A unified defense-in-depth layer that intercepts user queries before any processing:
    *   **PII Guardrail**: Detects sensitive data.
    *   **Access Authorization**: Validates the user's role against requested operations.
    *   **Content Safety Guard**: Blocks toxic inputs.
    *   **Prompt Injection Guard**: Detects adversarial jailbreak attempts.
    *   **Security Orchestrator**: Manages these guards, failing fast on malicious intent or triggering a Privacy Sanitizer to mask PII gracefully.

### 4. Query Processing & Enrichment (`QueryNLP/` and `query_enrichment.py`)
*   **Usage**: Run `python3 query_enrichment.py` (or other scripts in `QueryNLP/` for individual tests).
*   **Purpose**: Takes the sanitized user query and prepares it for the retrieval agent:
    *   **Intent Detection**: Routes the query based on classification.
    *   **Spelling & Entity Extraction**: Normalizes the text and pulls key terms.
    *   **Query Enrichment**: Expands the query semantically and factors in prior conversational chat history.
    *   **Query Embedding**: Vectorizes the finalized query for Qdrant retrieval.

---

## 🚀 Setup & Installation

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Vaibhav2005-r/Multi_Agent-Orchestration_with_RAG_-_Neurals.git
cd Multi_Agent-Orchestration_with_RAG_-_Neurals
pip install langchain-nvidia-ai-endpoints langchain-core langchain-qdrant qdrant-client pypdf pymupdf python-dotenv pandas torch transformers nemoguardrails nest_asyncio
```
> **Note for Windows Users:** The `nemoguardrails` dependency (specifically the `annoy` package) requires [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) to be installed.

### 2. Configure Environment Variables
Create a `.env` file in the root directory and add your API key from [build.nvidia.com](https://build.nvidia.com):
```env
NVIDIA_API_KEY=nvapi-your_api_key_here
```

---

## 📄 License
MIT License
