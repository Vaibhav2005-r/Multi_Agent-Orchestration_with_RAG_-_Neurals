"""
Flask API Server for Multi-Agent RAG Frontend
==============================================
Exposes the MasterQueryPipeline and Unified Ingestion Pipeline as REST endpoints.

Endpoints:
  GET  /api/health          — Health check
  POST /api/query           — Run the full end-to-end query pipeline
  GET  /api/pipeline-info    — Static info about pipeline stages and models
  POST /api/upload          — Upload document files (.pdf, .txt, .md) to staging
  POST /api/ingest          — Trigger the unified document + ingestion pipeline
  GET  /api/ingest/status   — Get real-time status of the ingestion pipeline
  GET  /api/documents       — List currently indexed documents and statistics
"""

import os
import sys
import time
import json
import shutil
import threading
from pathlib import Path
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
_pipeline_lock = threading.Lock()

def get_pipeline(force_reload=False):
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None or force_reload:
            from pipeline import MasterQueryPipeline
            _pipeline = MasterQueryPipeline()
    return _pipeline


# ── Ingestion Pipeline Live State ─────────────────────────────────────
_ingestion_state = {
    "status": "idle",             # "idle" | "running" | "completed" | "error"
    "current_stage": None,
    "stage_id": None,
    "stage_index": 0,
    "total_stages": 6,
    "progress": 0,
    "message": "Pipeline ready for document ingestion.",
    "processed_files": [],
    "chunks_count": 0,
    "elapsed_seconds": 0,
    "error": None
}
_state_lock = threading.Lock()

def update_ingestion_state(**kwargs):
    with _state_lock:
        _ingestion_state.update(kwargs)


# ─────────────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Multi-Agent RAG API is running."})


# ─────────────────────────────────────────────────────────────────────
# Pipeline Info (metadata for the visualizer)
# ─────────────────────────────────────────────────────────────────────
@app.route("/api/pipeline-info", methods=["GET"])
def pipeline_info():
    return jsonify({
        "ingestion_stages": [
            {
                "id": "loader",
                "name": "Document Loader",
                "description": "Scans uploaded files & Data/ directory for PDFs, TXTs & MDs, extracts raw text using PyMuPDF.",
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
                "name": "JSON/JSONL Export & Append",
                "description": "Exports and appends all enriched chunks to processed_documents.json and .jsonl artifacts.",
                "model": None,
                "icon": "download"
            },
            {
                "id": "indexing",
                "name": "Qdrant Vector Indexing",
                "description": "Bulk-inserts enriched chunks as dense vectors into the persistent Qdrant collection.",
                "model": "nvidia/llama-nemotron-embed-1b-v2",
                "icon": "database"
            },
            {
                "id": "verification",
                "name": "Verification Search & Reload",
                "description": "Runs test queries to confirm retrieval quality and reloads active vector store into memory.",
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
# Document Inventory Endpoint
# ─────────────────────────────────────────────────────────────────────
@app.route("/api/documents", methods=["GET"])
def list_documents():
    json_path = os.path.join(str(db_client.DATA_DIR), "processed_documents.json")
    if not os.path.exists(json_path):
        return jsonify({"total_chunks": 0, "sources": []})

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            docs = json.load(f)

        sources = {}
        for d in docs:
            meta = d.get("metadata", {})
            src = meta.get("filename") or meta.get("source") or "Unknown"
            sources[src] = sources.get(src, 0) + 1

        return jsonify({
            "total_chunks": len(docs),
            "sources": [{"filename": k, "chunks": v} for k, v in sources.items()]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
# Data Ingestion & Upload Endpoints
# ─────────────────────────────────────────────────────────────────────
def run_ingestion_background(skip_indexing=False, staging_dir="Data/upload_staging"):
    import asyncio
    from ingestion_pipeline import UnifiedIngestionPipeline

    update_ingestion_state(
        status="running",
        stage_id="loader",
        current_stage="Document Loader",
        stage_index=0,
        total_stages=6,
        progress=5,
        message="Starting document loading and parsing...",
        error=None
    )

    def on_progress(stage_id, stage_name, stage_idx, total_stages, message):
        pct = int(((stage_idx + 1) / (total_stages + 1)) * 100)
        update_ingestion_state(
            stage_id=stage_id,
            current_stage=stage_name,
            stage_index=stage_idx,
            total_stages=total_stages,
            progress=pct,
            message=message
        )

    try:
        # If staging directory has files, ingest from staging
        target_dir = staging_dir if os.path.exists(staging_dir) and os.listdir(staging_dir) else "Data"
        pipeline = UnifiedIngestionPipeline(
            data_dir=target_dir,
            skip_indexing=skip_indexing,
            progress_callback=on_progress
        )

        result = asyncio.run(pipeline.run())

        # Move staged files to Data/ directory
        if os.path.exists(staging_dir):
            for filename in os.listdir(staging_dir):
                src_path = os.path.join(staging_dir, filename)
                dst_path = os.path.join("Data", filename)
                try:
                    if os.path.exists(dst_path):
                        os.remove(dst_path)
                    shutil.move(src_path, dst_path)
                except Exception as e:
                    print(f"Error archiving file {filename}: {e}")

        # Reload in-memory query pipeline to immediately include new documents
        get_pipeline(force_reload=True)

        update_ingestion_state(
            status="completed",
            stage_id="completed",
            current_stage="Completed",
            stage_index=6,
            progress=100,
            message=f"Successfully ingested {result.get('chunks_count', 0)} chunks into vector store.",
            chunks_count=result.get("chunks_count", 0),
            processed_files=result.get("processed_files", []),
            elapsed_seconds=result.get("elapsed_seconds", 0)
        )

    except Exception as e:
        print(f"❌ Ingestion pipeline failed: {e}")
        update_ingestion_state(
            status="error",
            error=str(e),
            message=f"Pipeline error: {str(e)}"
        )


@app.route("/api/upload", methods=["POST"])
def upload_files():
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded."}), 400

    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No files selected."}), 400

    saved_files = []
    staging_dir = os.path.join(str(db_client.DATA_DIR), "upload_staging")
    os.makedirs(staging_dir, exist_ok=True)

    from werkzeug.utils import secure_filename
    for file in files:
        if file.filename:
            filename = secure_filename(file.filename)
            file.save(os.path.join(staging_dir, filename))
            saved_files.append(filename)

    return jsonify({
        "status": "success",
        "message": f"{len(saved_files)} file(s) staged successfully for combined ingestion.",
        "files": saved_files
    })


@app.route("/api/ingest", methods=["POST"])
def trigger_ingestion():
    with _state_lock:
        if _ingestion_state["status"] == "running":
            return jsonify({"status": "busy", "message": "An ingestion pipeline is already running."}), 409

    data = request.get_json(force=True, silent=True) or {}
    skip_indexing = data.get("skip_indexing", False)

    thread = threading.Thread(target=run_ingestion_background, args=(skip_indexing,))
    thread.daemon = True
    thread.start()

    return jsonify({
        "status": "started",
        "message": "Combined End-to-End Ingestion Pipeline started in background."
    })


@app.route("/api/ingest/status", methods=["GET"])
def get_ingest_status():
    with _state_lock:
        return jsonify(dict(_ingestion_state))


# ─────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Multi-Agent RAG — Flask API Server")
    print("  http://localhost:8080")
    print("============================================================")
    app.run(host="0.0.0.0", port=8080, debug=False)
