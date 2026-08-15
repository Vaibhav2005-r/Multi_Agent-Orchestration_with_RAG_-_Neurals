"""
Master End-to-End Query Pipeline
==================================
Connects every module in the system into a single, linear, end-to-end pipeline:

  [Raw User Query]
       │
       ▼
  [1] Query Processing Module   (query_processing/query_orchestrator.py)
      - Spelling Correction
      - Intent Classification
      - Entity Extraction
      - Semantic Expansion
       │
       │  enriched_query_payload
       ▼
  [2] Security Layer            (SecurityLayer/security_orchestrator.py)
      - Prompt Injection Check
      - Content Safety Check
      - RBAC Authorization Check
      - PII Detection & LLM Rewrite
       │
       │  safe_query  OR  BLOCKED (early exit)
       ▼
  [3] RAG Retrieval Layer       (rag_retrieval/master_orchestrator.py)
      - Secure Hybrid Search (BM25 + Qdrant)
      - Semantic Deduplication
      - Cascaded Reranking (FlashRank → NVIDIA Cross-Encoder)
      - Long-Context Reordering
       │
       │  context_string + source_documents
       ▼
  [4] Answer Synthesis Layer    (answer_synthesis/generator_llm.py)
      - Structured Prompt Construction
      - LLM Generation (NVIDIA Nemotron)
      - Hallucination Detection (Vectara)
      - Citation Injection & Follow-up Generation
       │
       ▼
  [Final Answer]

Usage:
  python pipeline.py
  python pipeline.py --query "What are the rules for loan disbursals?"
  python pipeline.py --role EMPLOYEE --query "NBFC compliance mandates"
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
import db_client  # Ensures Windows UTF-8 and path environment

load_dotenv()

# UTF-8 Stream Reconfiguration for Windows
if sys.platform == "win32":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# =====================================================================
# Master End-to-End Pipeline
# =====================================================================

class MasterQueryPipeline:
    """
    The top-level orchestrator that connects all four major subsystems:
    Query Processing → Security Layer → RAG Retrieval → Answer Synthesis
    """

    def __init__(self):
        print("\n" + "=" * 70)
        print("  INITIALIZING MASTER END-TO-END PIPELINE")
        print("=" * 70)

        # ── Module 1: Query Processing ───────────────────────────────
        print("\n[MODULE 1/4] Loading Query Processing Orchestrator...")
        from query_processing.query_orchestrator import QueryOrchestrator
        self.query_orchestrator = QueryOrchestrator()

        # ── Module 2: Security Layer ─────────────────────────────────
        # NOTE: Security is already embedded inside QueryEnricher (called by
        # QueryOrchestrator). Here we expose a dedicated standalone pass so
        # the security status is explicit and can gate the pipeline early.
        print("\n[MODULE 2/4] Loading Security Orchestrator...")
        from SecurityLayer.security_orchestrator import SecurityOrchestrator
        self.security_orchestrator = SecurityOrchestrator()

        # ── Module 3: RAG Retrieval ──────────────────────────────────
        print("\n[MODULE 3/4] Loading Master RAG Orchestrator...")
        from rag_retrieval.master_orchestrator import MasterRAGOrchestrator
        self.rag_orchestrator = MasterRAGOrchestrator()

        # ── Module 4: Answer Synthesis ───────────────────────────────
        print("\n[MODULE 4/4] Loading Answer Synthesis Generator...")
        from answer_synthesis.generator_llm import GeneratorLLM
        self.generator = GeneratorLLM()

        print("\n" + "=" * 70)
        print("  ✅  ALL MODULES LOADED — PIPELINE READY")
        print("=" * 70 + "\n")

    # ─────────────────────────────────────────────────────────────────
    # Stage 1: Query Processing
    # ─────────────────────────────────────────────────────────────────
    def _run_query_processing(
        self,
        raw_query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_role: str = "EMPLOYEE"
    ) -> Dict[str, Any]:
        """
        Stage 1: Spelling correction, intent detection, entity extraction,
        and semantic enrichment of the raw user query.
        """
        print("\n" + "─" * 60)
        print("  STAGE 1 — Query Processing")
        print("─" * 60)
        t0 = time.time()
        result = self.query_orchestrator.process(
            query=raw_query,
            chat_history=chat_history or [],
            user_role=user_role
        )
        print(f"\n  ✅ Stage 1 done in {time.time()-t0:.2f}s")
        return result

    # ─────────────────────────────────────────────────────────────────
    # Stage 2: Security Gate
    # ─────────────────────────────────────────────────────────────────
    def _run_security_check(
        self,
        query: str,
        user_role: str
    ) -> Dict[str, Any]:
        """
        Stage 2: Explicit security evaluation gate on the raw query.
        
        Runs the security checks concurrently.
        
        Returns a security result dict:
          { "status": "ALLOW" | "BLOCK", "query": <safe_query>, "reason": <if BLOCK> }
        """
        print("\n" + "─" * 60)
        print("  STAGE 2 — Security Layer")
        print("─" * 60)
        t0 = time.time()

        print(f"  → Evaluating security for query: '{query}' (Role: {user_role})")
        security_result = self.security_orchestrator.evaluate_query(
            query=query,
            user_role=user_role
        )

        status = security_result.get("status", "ALLOW")
        safe_query = security_result.get("query", query)

        if status == "BLOCK":
            reason = security_result.get("reason", "Unknown security violation.")
            print(f"\n  🚫 QUERY BLOCKED: {reason}")
        else:
            print(f"\n  ✅ Security check PASSED in {time.time()-t0:.2f}s")
            if safe_query != query:
                print(f"  ℹ️  Query sanitized (PII removed): '{safe_query}'")

        return {
            "status": status,
            "safe_query": safe_query,
            "reason": security_result.get("reason", "")
        }

    # ─────────────────────────────────────────────────────────────────
    # Stage 3: RAG Retrieval
    # ─────────────────────────────────────────────────────────────────
    def _run_rag_retrieval(self, safe_query: str) -> Dict[str, Any]:
        """
        Stage 3: Hybrid retrieval, deduplication, cascaded reranking and
        context packaging against the Qdrant vector database.

        Receives the security-cleared query and returns the assembled
        context string + source document list ready for the LLM.
        """
        print("\n" + "─" * 60)
        print("  STAGE 3 — RAG Retrieval")
        print("─" * 60)
        t0 = time.time()

        # Use the enriched rewritten query for retrieval if available,
        # otherwise fall back to the security-cleared cleaned query
        retrieval_result = self.rag_orchestrator.run_pipeline(safe_query)

        elapsed = time.time() - t0
        status = retrieval_result.get("status", "UNKNOWN")
        n_docs = len(retrieval_result.get("documents", []))

        if status == "BLOCKED_OR_EMPTY":
            print(f"\n  ⚠️  RAG pipeline returned no documents.")
        else:
            print(f"\n  ✅ Stage 3 done in {elapsed:.2f}s — {n_docs} documents retrieved.")

        return retrieval_result

    # ─────────────────────────────────────────────────────────────────
    # Stage 4: Answer Synthesis
    # ─────────────────────────────────────────────────────────────────
    def _run_answer_synthesis(
        self,
        query_result: Dict[str, Any],
        retrieval_result: Dict[str, Any]
    ) -> str:
        """
        Stage 4: Builds the structured LLM prompt, generates the answer,
        runs hallucination detection, injects citations, and returns the
        final formatted response.

        The query_result from Stage 1 (intent, entities, enriched payload)
        and the retrieval_result from Stage 3 (context_string, documents)
        are both passed to the prompt constructor.
        """
        print("\n" + "─" * 60)
        print("  STAGE 4 — Answer Synthesis")
        print("─" * 60)
        t0 = time.time()

        final_answer = self.generator.generate_answer(
            query_processing_result=query_result,
            retrieval_result=retrieval_result
        )

        print(f"\n  ✅ Stage 4 done in {time.time()-t0:.2f}s")
        return final_answer

    # ─────────────────────────────────────────────────────────────────
    # Main Pipeline Entry Point
    # ─────────────────────────────────────────────────────────────────
    def run(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_role: str = "EMPLOYEE"
    ) -> Dict[str, Any]:
        """
        End-to-end pipeline entry point.

        Args:
            query:        The raw user query string.
            chat_history: Optional list of prior conversation turns.
            user_role:    RBAC role for the user (EMPLOYEE | ADMIN).

        Returns:
            A dict with keys:
              - "final_answer":     The fully synthesized, cited answer string.
              - "query_result":     Output from Stage 1 (Query Processing).
              - "security_result":  Output from Stage 2 (Security check).
              - "retrieval_result": Output from Stage 3 (RAG Retrieval).
              - "blocked":          True if the query was blocked by security.
        """
        t_pipeline_start = time.time()
        print("\n" + "=" * 70)
        print(f"  PIPELINE START: '{query}'")
        print(f"  User Role: {user_role}")
        print("=" * 70)

        # ── Stages 1 & 2: Concurrent Query Processing and Security ───
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_query = executor.submit(self._run_query_processing, query, chat_history, user_role)
            future_security = executor.submit(self._run_security_check, query, user_role)
            
            query_result = future_query.result()
            security_result = future_security.result()

        if security_result["status"] == "BLOCK":
            total_time = time.time() - t_pipeline_start
            blocked_answer = (
                f"⛔ **Your query has been blocked by the security layer.**\n\n"
                f"**Reason:** {security_result['reason']}\n\n"
                f"Please rephrase your query and try again."
            )
            print("\n" + "=" * 70)
            print(f"  ⛔ PIPELINE BLOCKED in {total_time:.2f}s")
            print("=" * 70)
            return {
                "final_answer": blocked_answer,
                "query_result": query_result,
                "security_result": security_result,
                "retrieval_result": {},
                "blocked": True
            }

        # Inject the security-sanitized query into the query_result so that
        # the prompt constructor uses the safe, clean version
        safe_query = security_result["safe_query"]
        if query_result.get("enriched_payload"):
            query_result["enriched_payload"]["query"] = safe_query
        query_result["cleaned_query"] = safe_query

        # ── Stage 3: RAG Retrieval ───────────────────────────────────
        retrieval_result = self._run_rag_retrieval(safe_query)

        # ── Stage 4: Answer Synthesis ────────────────────────────────
        final_answer = self._run_answer_synthesis(query_result, retrieval_result)

        total_time = time.time() - t_pipeline_start
        print("\n" + "=" * 70)
        print(f"  ✅ PIPELINE COMPLETE in {total_time:.2f}s")
        print("=" * 70)

        return {
            "final_answer": final_answer,
            "query_result": query_result,
            "security_result": security_result,
            "retrieval_result": retrieval_result,
            "blocked": False
        }

    def cleanup(self):
        """Release underlying resources cleanly."""
        try:
            self.rag_orchestrator.cleanup()
        except Exception:
            pass


# =====================================================================
# CLI Entry Point
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Master End-to-End Query Pipeline"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="What are the rules regarding loan disbursals and fees paid to LSPs?",
        help="The user query to process"
    )
    parser.add_argument(
        "--role", "-r",
        type=str,
        default="EMPLOYEE",
        choices=["EMPLOYEE", "ADMIN"],
        help="User RBAC role (default: EMPLOYEE)"
    )
    parser.add_argument(
        "--history",
        type=str,
        default=None,
        help="Optional JSON string of chat history: '[{\"role\":\"user\",\"content\":\"...\"}]'"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    chat_history = None
    if args.history:
        try:
            chat_history = json.loads(args.history)
        except json.JSONDecodeError:
            print("[Warning] Could not parse --history as JSON. Ignoring.")

    pipeline = MasterQueryPipeline()

    result = pipeline.run(
        query=args.query,
        chat_history=chat_history,
        user_role=args.role
    )

    print("\n" + "=" * 70)
    print("  FINAL SYNTHESIZED ANSWER")
    print("=" * 70)
    # Safe encode for terminals that don't support all unicode chars
    print(result["final_answer"].encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding))

    pipeline.cleanup()
