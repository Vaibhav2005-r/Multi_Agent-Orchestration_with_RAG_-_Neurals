"""
Unified Data Ingestion Pipeline
================================
A single-entrypoint pipeline that connects Document Processing (loading, semantic chunking,
metadata enrichment, artifact export) directly to Qdrant Vector Indexing and Verification.

Workflow:
  [Upload / Data/] → [1. Document Loader] → [2. Semantic Chunking] → [3. LLM Metadata Enrichment]
                   → [4. Export JSON/JSONL] → [5. Qdrant Vector Indexing] → [6. Verification Search]

Usage:
  python ingestion_pipeline.py
  python ingestion_pipeline.py --data-dir Data/upload_staging --collection fintech_documents
"""

import os
import sys
import time
import asyncio
import argparse
from typing import List, Optional, Callable, Dict, Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from db_client import DEFAULT_QDRANT_PATH, DEFAULT_COLLECTION, DATA_DIR

# Load env first
load_dotenv()

# UTF-8 Stream Reconfiguration for Windows
if sys.platform == "win32":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# =====================================================================
# Pipeline modules
# =====================================================================
from document_pipeline import DocumentPipeline
from qdrant_indexer import QdrantIndexer


# =====================================================================
# Unified Ingestion Orchestrator
# =====================================================================

class UnifiedIngestionPipeline:
    """
    Connects the Document Processing Pipeline to the Qdrant Indexing Pipeline
    in a single seamless execution. Documents flow in-memory from enrichment
    directly to Qdrant and are persisted to JSON/JSONL.
    """

    def __init__(
        self,
        data_dir: str = str(DATA_DIR),
        qdrant_path: str = DEFAULT_QDRANT_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2",
        batch_size: int = 100,
        skip_indexing: bool = False,
        progress_callback: Optional[Callable[[str, str, int, int, str], None]] = None
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.qdrant_path = os.path.abspath(qdrant_path)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.skip_indexing = skip_indexing
        self.progress_callback = progress_callback

        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY must be set in .env or environment variables.")

        print("=" * 80)
        print("  UNIFIED DATA INGESTION PIPELINE")
        print(f"  Data Dir: {self.data_dir} | Qdrant: {self.qdrant_path}")
        print(f"  Collection: {self.collection_name}")
        print("=" * 80)

        # Initialize Document Processing Pipeline
        self.doc_pipeline = DocumentPipeline(
            api_key=api_key,
            llm_model="meta/llama-3.1-8b-instruct",
            embedding_model="nvidia/nv-embedqa-e5-v5",
            batch_size=5,
            max_rpm=40,
        )

        self.qdrant_indexer: Optional[QdrantIndexer] = None

    def _notify(self, stage_id: str, stage_name: str, stage_idx: int, total_stages: int, message: str):
        if self.progress_callback:
            try:
                self.progress_callback(stage_id, stage_name, stage_idx, total_stages, message)
            except Exception as e:
                print(f"[Callback Warning] {e}")

    async def run(self) -> Dict[str, Any]:
        """Executes the full unified pipeline end-to-end with granular stage reporting."""
        t_start = time.time()
        total_stages = 6 if not self.skip_indexing else 4

        # ── Stage 1: Document Loading ─────────────────────────────────
        self._notify("loader", "Document Loader", 0, total_stages, f"Scanning and loading files from {self.data_dir}...")
        print("\n" + "─" * 60)
        print("  STAGE 1: Document Loading & Parsing")
        print("─" * 60)
        raw_docs = self.doc_pipeline.load_data(self.data_dir)
        if not raw_docs:
            print("[Pipeline] No documents found in target directory. Halting.")
            self._notify("loader", "Document Loader", 0, total_stages, "No documents found.")
            return {"status": "empty", "chunks_count": 0, "processed_files": []}

        loaded_files = list({doc.metadata.get("filename", "unknown") for doc in raw_docs})
        print(f"Loaded {len(raw_docs)} document items from: {loaded_files}")

        # ── Stage 2: Document Metadata & Semantic Chunking ────────────
        self._notify("chunking", "Semantic Chunking", 1, total_stages, f"Extracting metadata and semantic chunking for {len(loaded_files)} file(s)...")
        print("\n" + "─" * 60)
        print("  STAGE 2: Document-Level Metadata & Semantic Chunking")
        print("─" * 60)
        doc_meta_map = await self.doc_pipeline.extract_document_metadata_async(raw_docs)
        semantic_chunks = self.doc_pipeline.semantic_chunking(raw_docs, doc_meta_map)
        print(f"Generated {len(semantic_chunks)} semantic chunks.")

        # ── Stage 3: LLM Metadata Enrichment ───────────────────────────
        self._notify("enrichment", "Metadata Enrichment", 2, total_stages, f"Enriching {len(semantic_chunks)} chunks via Llama-3.1-8B (rate-limited)...")
        print("\n" + "─" * 60)
        print("  STAGE 3: Chunk-Level LLM Metadata Enrichment")
        print("─" * 60)
        enriched_docs = await self.doc_pipeline.enrich_chunks_async(semantic_chunks)

        # ── Stage 4: Artifact Export & Append ─────────────────────────
        self._notify("export", "JSON/JSONL Export", 3, total_stages, "Exporting and appending enriched chunks to Data/processed_documents.json...")
        print("\n" + "─" * 60)
        print("  STAGE 4: Exporting Enriched Chunks to JSON & JSONL")
        print("─" * 60)
        self.doc_pipeline.export_processed_data(append=True)

        # ── Stage 5: Qdrant Indexing ──────────────────────────────────
        if not self.skip_indexing:
            self._notify("indexing", "Qdrant Vector Indexing", 4, total_stages, f"Indexing {len(enriched_docs)} vectors into Qdrant '{self.collection_name}'...")
            print("\n" + "─" * 60)
            print("  STAGE 5: Qdrant Vector Indexing")
            print("─" * 60)
            self.qdrant_indexer = QdrantIndexer(
                collection_name=self.collection_name,
                embedding_model=self.embedding_model,
                qdrant_path=self.qdrant_path,
                batch_size=self.batch_size,
            )
            self.qdrant_indexer.index_documents(enriched_docs)

            # ── Stage 6: Verification Search ─────────────────────────
            self._notify("verification", "Verification Search", 5, total_stages, "Running verification search query...")
            print("\n" + "─" * 60)
            print("  STAGE 6: Verification Search")
            print("─" * 60)
            test_query = "What are the key rules and compliance requirements?"
            res = self.qdrant_indexer.search(test_query, k=2)
            print(f"Verification query completed: {len(res)} results returned.")

        total_time = round(time.time() - t_start, 2)
        self._notify("completed", "Ingestion Complete", total_stages, total_stages, f"Successfully processed {len(enriched_docs)} chunks in {total_time}s.")

        return {
            "status": "completed",
            "chunks_count": len(enriched_docs),
            "processed_files": loaded_files,
            "elapsed_seconds": total_time
        }


# =====================================================================
# CLI Entry Point
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Data Ingestion Pipeline — Document Processing → Qdrant Indexing"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="Data",
        help="Path to the data directory containing raw PDFs/TXTs"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help="Qdrant collection name"
    )
    parser.add_argument(
        "--qdrant-path",
        type=str,
        default=DEFAULT_QDRANT_PATH,
        help="Path for local Qdrant database"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of documents per Qdrant indexing batch"
    )
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="If set, only runs document processing & export — skips Qdrant indexing"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pipeline = UnifiedIngestionPipeline(
        data_dir=args.data_dir,
        qdrant_path=args.qdrant_path,
        collection_name=args.collection,
        batch_size=args.batch_size,
        skip_indexing=args.skip_indexing,
    )

    asyncio.run(pipeline.run())
