"""
Flask API Server for Multi-Agent RAG Frontend
==============================================
Exposes the MasterQueryPipeline as REST endpoints for the React frontend.

Endpoints:
  GET  /api/health       — Health check
  POST /api/query        — Run the full end-to-end query pipeline
  GET  /api/pipeline-info — Static info about pipeline stages and models

Usage:
  pip install flask flask-cors
  python api_server.py
"""

import os
import sys
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import db_client  # Enforces cross-platform path resolution and UTF-8 output

load_dotenv()

# UTF-8 Stream Reconfiguration for Windows
if sys.platform == "win32":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

app = Flask(__name__)
CORS(app)  # Allow React dev server to call this API

# ── Lazy-initialize the heavy pipeline once on first request ──────────
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from pipeline import MasterQueryPipeline
        _pipeline = MasterQueryPipeline()
    return _pipeline


# ─────────────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Multi-Agent RAG API is running."})


# ─────────────────────────────────────────────────────────────────────
# Pipeline Info (static metadata for the visualizer)
# ─────────────────────────────────────────────────────────────────────
@app.route("/api/pipeline-info", methods=["GET"])
def pipeline_info():
    return jsonify({
        "ingestion_stages": [
            {
                "id": "loader",
                "name": "Document Loader",
                "description": "Scans Data/ directory for PDFs & TXT files, extracts raw text using PyMuPDF.",
                "model": None,
                "icon": "file-text"
            },
            {
                "id": "chunking",
                "name": "Semantic Chunking",
                "description": "Splits documents at semantic breakpoints using cosine similarity of sentence embeddings.",
                "model": "nvidia/nv-embedqa-e5-v5",
                "icon": "scissors"
            },
            {
                "id": "enrichment",
                "name": "Metadata Enrichment",
                "description": "Extracts chunk title, entities, QA pairs and compliance mandates via async LLM batch calls (≤40 RPM).",
                "model": "meta/llama-3.1-8b-instruct",
                "icon": "sparkles"
            },
            {
                "id": "export",
                "name": "JSON/JSONL Export",
                "description": "Exports all enriched chunks to processed_documents.json and .jsonl artifacts.",
                "model": None,
                "icon": "download"
            },
            {
                "id": "indexing",
                "name": "Qdrant Vector Indexing",
                "description": "Bulk-inserts enriched chunks as dense vectors into a persistent Qdrant collection.",
                "model": "nvidia/llama-nemotron-embed-1b-v2",
                "icon": "database"
            },
            {
                "id": "verification",
                "name": "Verification Search",
                "description": "Runs test queries against the freshly indexed collection to confirm retrieval quality.",
                "model": None,
                "icon": "check-circle"
            }
        ],
        "query_stages": [
            {
                "id": "query_processing",
                "name": "Query Processing",
                "description": "Spelling correction, intent classification (PyTorch RNN), entity extraction, and semantic expansion.",
                "models": ["meta/llama-3.1-8b-instruct", "distilbert-base-uncased + BiLSTM", "nvidia/llama-nemotron-embed-1b-v2"],
                "icon": "search"
            },
            {
                "id": "security",
                "name": "Security Layer",
                "description": "Prompt injection detection, content safety check, RBAC authorization, and PII masking via LLM rewrite.",
                "models": ["nvidia/llama-3.1-nemoguard-8b-content-safety", "meta/llama-3.1-8b-instruct"],
                "icon": "shield"
            },
            {
                "id": "rag_retrieval",
                "name": "RAG Retrieval",
                "description": "Hybrid BM25 + Qdrant search (RRF fusion), semantic deduplication, cascaded reranking (FlashRank → NVIDIA Cross-Encoder), LongContextReorder.",
                "models": ["nvidia/llama-nemotron-rerank-1b-v2", "ms-marco-MultiBERT-L-12 (FlashRank)"],
                "icon": "layers"
            },
            {
                "id": "answer_synthesis",
                "name": "Answer Synthesis",
                "description": "Structured prompt construction, LLM generation with executive tone enforcement, hallucination detection, citation injection, and follow-up generation.",
                "models": ["nvidia/nemotron-3-super-120b-a12b", "vectara/hallucination_evaluation_model"],
                "icon": "cpu"
            }
        ]
    })


# ─────────────────────────────────────────────────────────────────────
# Main Query Endpoint
# ─────────────────────────────────────────────────────────────────────
@app.route("/api/query", methods=["POST"])
def run_query():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    role = data.get("role", "GUEST").upper()
    chat_history = data.get("chat_history", [])

    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    if role not in ("GUEST", "EMPLOYEE", "ADMIN"):
        role = "GUEST"

    try:
        pipeline = get_pipeline()
        t_start = time.time()
        result = pipeline.run(query=query, chat_history=chat_history, user_role=role)
        elapsed = round(time.time() - t_start, 2)

        # Build a clean, serializable response
        query_r = result.get("query_result", {})
        security_r = result.get("security_result", {})
        retrieval_r = result.get("retrieval_result", {})

        return jsonify({
            "final_answer": result.get("final_answer", ""),
            "blocked": result.get("blocked", False),
            "elapsed_seconds": elapsed,
            "stages": {
                "query_processing": {
                    "original_query": query_r.get("original_query", query),
                    "cleaned_query": query_r.get("cleaned_query", query),
                    "intent": query_r.get("intent", "UNKNOWN"),
                    "entities": query_r.get("entities", []),
                    "enriched_query": query_r.get("enriched_payload", {}).get("query", query)
                },
                "security": {
                    "status": security_r.get("status", "ALLOW"),
                    "safe_query": security_r.get("safe_query", query),
                    "reason": security_r.get("reason", "")
                },
                "rag_retrieval": {
                    "status": retrieval_r.get("status", "UNKNOWN"),
                    "num_documents": len(retrieval_r.get("documents", [])),
                    "sources": list({
                        doc.metadata.get("source", "Unknown")
                        for doc in retrieval_r.get("documents", [])
                    }) if retrieval_r.get("documents") else []
                },
                "answer_synthesis": {
                    "model": "nvidia/nemotron-3-super-120b-a12b",
                    "hallucination_model": "vectara/hallucination_evaluation_model"
                }
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Multi-Agent RAG — Flask API Server")
    print("  http://localhost:8080")
    print("============================================================")
    app.run(host="0.0.0.0", port=8080, debug=False)
