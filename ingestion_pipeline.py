"""
Unified Data Ingestion Pipeline
================================
A single-entrypoint pipeline that connects the Document Processing Pipeline
directly to the Qdrant Indexing Pipeline — no manual intermediate steps.

Workflow:
  [Data/] → Document Loading → Metadata Enrichment → Export JSON/JSONL
          → In-Memory Handoff → Qdrant Indexing → Verification Search

Usage:
  python ingestion_pipeline.py
  python ingestion_pipeline.py --data-dir Data/ --collection my_collection
  python ingestion_pipeline.py --skip-indexing   # Only run document processing
"""

import os
import sys
import time
import asyncio
import argparse
from typing import List, Optional

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
# Inline import of both pipeline modules
# =====================================================================
from document_pipeline import DocumentPipeline
from qdrant_indexer import QdrantIndexer


# =====================================================================
# Unified Ingestion Orchestrator
# =====================================================================

class UnifiedIngestionPipeline:
    """
    Connects the Document Processing Pipeline to the Qdrant Indexing Pipeline
    in a single seamless execution. Documents are handed off in-memory from
    enrichment directly to Qdrant — no JSON round-trip required for indexing.
    """

    def __init__(
        self,
        data_dir: str = str(DATA_DIR),
        qdrant_path: str = DEFAULT_QDRANT_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2",
        batch_size: int = 100,
        skip_indexing: bool = False
    ):
        self.data_dir = data_dir
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.skip_indexing = skip_indexing

        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY must be set in .env or environment variables.")

        print("=" * 80)
        print("  UNIFIED DATA INGESTION PIPELINE")
        print(f"  Data Dir: {self.data_dir} | Qdrant: {self.qdrant_path}")
        print(f"  Collection: {self.collection_name}")
        print("=" * 80)

        # ── Stage 1: Document Processing Pipeline ──────────────────────
        print("\n[STAGE 1] Initializing Document Processing Pipeline...")
        self.doc_pipeline = DocumentPipeline(
            api_key=api_key,
            llm_model="meta/llama-3.1-8b-instruct",
            embedding_model="nvidia/nv-embedqa-e5-v5",  # Used for semantic chunking
            batch_size=5,
            max_rpm=40,
        )

        # ── Stage 2: Qdrant Indexing Pipeline (lazy-initialized after enrichment) ──
        self.qdrant_indexer: Optional[QdrantIndexer] = None

    async def _run_document_stage(self) -> List[Document]:
        """Runs the document loading, enrichment, and export stage."""
        print("\n" + "─" * 60)
        print("  STAGE 1: Document Processing & Enrichment")
        print("─" * 60)

        enriched_docs = await self.doc_pipeline.run_pipeline_async(data_dir=self.data_dir)

        if not enriched_docs:
            print("[Pipeline] No documents were enriched. Halting.")
            return []

        print(f"\n✅ Stage 1 Complete — {len(enriched_docs)} enriched chunks ready.")
        return enriched_docs

    def _run_indexing_stage(self, enriched_docs: List[Document]):
        """
        Runs the Qdrant indexing stage using in-memory documents directly.
        No JSON round-trip — documents passed straight from Stage 1.
        """
        print("\n" + "─" * 60)
        print("  STAGE 2: Qdrant Vector Indexing")
        print("─" * 60)
        print(f"  → Indexing {len(enriched_docs)} chunks into '{self.collection_name}'...")

        self.qdrant_indexer = QdrantIndexer(
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
            qdrant_path=self.qdrant_path,
            batch_size=self.batch_size,
        )

        # Direct in-memory handoff — no load_documents() needed
        self.qdrant_indexer.index_documents(enriched_docs)
        print(f"\n✅ Stage 2 Complete — Qdrant collection '{self.collection_name}' is up to date.")

    def _run_verification(self):
        """Runs a test query against the freshly indexed Qdrant collection."""
        print("\n" + "─" * 60)
        print("  STAGE 3: Verification Search")
        print("─" * 60)

        test_queries = [
            "What are the rules regarding loan disbursals and fees paid to LSPs?",
            "What are the compliance mandates for NBFCs?",
        ]

        for query in test_queries:
            print(f"\n  🔍 Query: '{query}'")
            results = self.qdrant_indexer.search(query, k=2)
            for i, res in enumerate(results, 1):
                title = res.metadata.get("chunk_title", "N/A")
                source = res.metadata.get("source", "N/A")
                print(f"    Result {i}: [{title}] — {source}")
                print(f"    Preview : {res.page_content[:120].replace(chr(10), ' ')}...")

        print(f"\n✅ Stage 3 Complete — Verification search successful.")

    async def run(self):
        """Executes the full unified pipeline end-to-end."""
        t_start = time.time()

        # ── Stage 1: Document Processing ──
        enriched_docs = await self._run_document_stage()
        if not enriched_docs:
            return

        # ── Stage 2: Qdrant Indexing (optional skip for quick testing) ──
        if not self.skip_indexing:
            self._run_indexing_stage(enriched_docs)

            # ── Stage 3: Verification ──
            self._run_verification()
        else:
            print("\n[Skip] Qdrant indexing skipped (--skip-indexing flag set).")

        total = time.time() - t_start
        print("\n" + "=" * 80)
        print(f"  ✅ UNIFIED INGESTION PIPELINE COMPLETE in {total:.2f}s")
        print(f"     Processed  : {len(enriched_docs)} chunks")
        print(f"     Exported   : Data/processed_documents.json & .jsonl")
        if not self.skip_indexing:
            print(f"     Indexed to : {self.qdrant_path} / {self.collection_name}")
        print("=" * 80)


# =====================================================================
# Entry Point
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Data Ingestion Pipeline — Document Processing → Qdrant Indexing"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="Data",
        help="Path to the data directory containing raw PDFs/TXTs (default: Data)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="fintech_documents_optimized",
        help="Qdrant collection name (default: fintech_documents_optimized)"
    )
    parser.add_argument(
        "--qdrant-path",
        type=str,
        default="Data/qdrant_db_optimized",
        help="Path for local Qdrant database (default: Data/qdrant_db_optimized)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of documents per Qdrant indexing batch (default: 100)"
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
