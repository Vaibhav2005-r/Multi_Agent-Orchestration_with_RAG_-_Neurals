"""
Document Processing & Enrichment Pipeline powered by NVIDIA AI Endpoints.

Pipeline Workflow:
1. Document Loading: Scans and loads documents (PDF, MD, TXT, etc.) from data folder with file-level metadata.
2. Document-Level Metadata Extraction: Extracts high-level metadata (doc type, summary, jurisdiction, entities) via ChatNVIDIA.abatch().
3. Semantic Chunking: Splits text based on semantic similarity breakpoints using NVIDIAEmbeddings.
4. Chunk-Level Metadata Enrichment: Extracts granular metadata (title, summary, entities, topics, compliance rules, QA pairs) via ChatNVIDIA.abatch() strictly throttled to <= 40 RPM.
5. Vector Storage & Export: Indexes enriched chunks into an InMemoryVectorStore and exports to JSON/JSONL.
"""

import os
import glob
import json
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import deque

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# Load environment variables
load_dotenv()


# =====================================================================
# Asynchronous Rate Limiter (Strict RPM Enforcer & Token Bucket)
# =====================================================================

class AsyncRateLimiter:
    """
    Sliding-window asynchronous rate limiter.
    Ensures that at most `max_rpm` requests are dispatched in any 60-second window.
    """
    def __init__(self, max_rpm: int = 40):
        self.max_rpm = max(1, max_rpm)
        self.timestamps: deque = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, num_requests: int = 1) -> float:
        """
        Acquires permission to fire `num_requests`.
        If the current 60s sliding window is full, sleeps until tokens expire.
        Returns the duration waited in seconds.
        """
        total_waited = 0.0
        async with self._lock:
            while True:
                now = time.time()
                # Discard timestamps older than 60 seconds
                while self.timestamps and (now - self.timestamps[0]) >= 60.0:
                    self.timestamps.popleft()

                # Check if adding num_requests fits in the current window
                if len(self.timestamps) + num_requests <= self.max_rpm:
                    for _ in range(num_requests):
                        self.timestamps.append(time.time())
                    break
                else:
                    # Calculate how long to wait until enough oldest requests exit the 60s window
                    oldest = self.timestamps[0]
                    sleep_time = max(0.2, 60.0 - (now - oldest) + 0.1)
                    total_waited += sleep_time
                    await asyncio.sleep(sleep_time)

        return total_waited

    @property
    def current_rpm_usage(self) -> int:
        """Returns the number of requests sent in the last 60 seconds."""
        now = time.time()
        while self.timestamps and (now - self.timestamps[0]) >= 60.0:
            self.timestamps.popleft()
        return len(self.timestamps)


# =====================================================================
# Pydantic Schemas for Structured Metadata Extraction
# =====================================================================

class DocumentOverviewMetadata(BaseModel):
    """Document-level high-level metadata."""
    document_title: str = Field(description="Official or deduced title of the entire document")
    document_type: str = Field(description="Type of document, e.g., Regulatory Guideline, Policy, Financial Report, Whitepaper")
    issuing_authority: str = Field(description="Organization or authority that issued the document, e.g., RBI, SEC, Company Name")
    effective_date: str = Field(description="Date or time period mentioned in document, or 'Unknown'")
    primary_domain: str = Field(description="Primary domain/sector, e.g., Digital Lending, Fintech, Banking, Corporate Finance")
    executive_summary: str = Field(description="Comprehensive 2-4 sentence executive overview of the document purpose and scope")
    key_stakeholders: List[str] = Field(description="Main stakeholders targeted or regulated, e.g., Regulated Entities, Borrowers, LSPs")


class EnrichedChunkMetadata(BaseModel):
    """Chunk-level granular semantic metadata."""
    chunk_title: str = Field(description="Concise, descriptive title for the specific content in this chunk")
    chunk_summary: str = Field(description="1-2 sentence summary capturing key facts and context of this chunk")
    entities: List[str] = Field(description="Key financial/legal entities, institutions, acronyms (e.g. RE, LSP, DLA, APR, KFS)")
    topics: List[str] = Field(description="Top 2-4 specific topics or keywords covered in this chunk")
    compliance_mandates: List[str] = Field(description="Actionable rules, obligations, restrictions, or requirements in this chunk")
    potential_questions: List[str] = Field(description="2-3 realistic user questions that can be answered directly using this chunk")


# =====================================================================
# Document Loader
# =====================================================================

class DocumentLoader:
    """Discovers and loads documents from directory with base file metadata."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
    IGNORE_FILES = {"processed_documents.json", "processed_documents.jsonl"}

    @classmethod
    def load_pdf(cls, file_path: str) -> List[Document]:
        """Extract text from PDF using PyMuPDF (fitz) or pdfplumber."""
        docs = []
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text().strip()
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "filename": Path(file_path).name,
                            "page": page_num + 1,
                            "total_pages": len(doc),
                            "file_type": "pdf"
                        }
                    ))
            doc.close()
            return docs
        except Exception as e:
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if text and text.strip():
                            docs.append(Document(
                                page_content=text.strip(),
                                metadata={
                                    "source": file_path,
                                    "filename": Path(file_path).name,
                                    "page": page_num + 1,
                                    "total_pages": len(pdf.pages),
                                    "file_type": "pdf"
                                }
                            ))
                return docs
            except Exception as e2:
                print(f"[DocumentLoader] Error loading PDF {file_path}: {e2}")
                return []

    @classmethod
    def load_text(cls, file_path: str) -> List[Document]:
        """Load text or markdown file."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
            if content:
                return [Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "filename": Path(file_path).name,
                        "page": 1,
                        "total_pages": 1,
                        "file_type": Path(file_path).suffix.lstrip(".")
                    }
                )]
        except Exception as e:
            print(f"[DocumentLoader] Error loading text file {file_path}: {e}")
        return []

    @classmethod
    def load_directory(cls, data_dir: str) -> List[Document]:
        """Recursively loads all supported documents from directory."""
        all_docs: List[Document] = []
        path = Path(data_dir)
        if not path.exists():
            print(f"[DocumentLoader] Directory '{data_dir}' does not exist.")
            return []

        for file_path in path.rglob("*"):
            if file_path.is_file():
                if file_path.name in cls.IGNORE_FILES:
                    continue
                if file_path.suffix.lower() in cls.SUPPORTED_EXTENSIONS:
                    suffix = file_path.suffix.lower()
                    if suffix == ".pdf":
                        loaded = cls.load_pdf(str(file_path))
                    else:
                        loaded = cls.load_text(str(file_path))
                    print(f"  -> Loaded '{file_path.name}': {len(loaded)} page(s)/section(s)")
                    all_docs.extend(loaded)

        return all_docs


# =====================================================================
# Document Pipeline with NVIDIA Models & 40 RPM Rate Limiter
# =====================================================================

class DocumentPipeline:
    """End-to-end Document Pipeline leveraging NVIDIA Builder Models with strict RPM throttling."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        llm_model: str = "meta/llama-3.1-8b-instruct",
        embedding_model: str = "nvidia/nv-embedqa-e5-v5",
        temperature: float = 0.1,
        max_completion_tokens: int = 1024,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: Optional[float] = 85.0,
        batch_size: int = 5,
        max_rpm: int = 40,
        **kwargs
    ):
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY must be provided or set in environment variables.")

        self.batch_size = batch_size
        self.max_rpm = max_rpm
        self.llm_model_name = llm_model
        self.embedding_model_name = embedding_model

        # Dedicated Async Rate Limiter
        self.rate_limiter = AsyncRateLimiter(max_rpm=self.max_rpm)

        # Initialize NVIDIA LLM
        self.llm = ChatNVIDIA(
            model=llm_model,
            api_key=self.api_key,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            timeout=120,
        )

        # Initialize NVIDIA Embeddings
        self.embeddings = NVIDIAEmbeddings(
            model=embedding_model,
            api_key=self.api_key,
            truncate="END",
        )

        # Initialize Semantic Chunker with NVIDIA Embeddings
        self.semantic_chunker = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )

        # Parsers and Prompt Templates
        self.doc_parser = JsonOutputParser(pydantic_object=DocumentOverviewMetadata)
        self.doc_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert document intelligence assistant. Analyze the document excerpt and extract structured high-level metadata in JSON matching the exact schema:\n{format_instructions}"
            ),
            (
                "user",
                "Document Name: {filename}\nFile Path: {source}\n\nDocument Content Excerpt:\n{content}\n\nExtract document overview metadata:"
            )
        ]).partial(format_instructions=self.doc_parser.get_format_instructions())
        self.doc_chain = self.doc_prompt | self.llm | self.doc_parser

        self.chunk_parser = JsonOutputParser(pydantic_object=EnrichedChunkMetadata)
        self.chunk_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert financial and compliance metadata extraction model. Analyze the chunk and extract enriched metadata in JSON matching this schema:\n{format_instructions}"
            ),
            (
                "user",
                "Document: {document_title}\nDomain: {primary_domain}\nSource File: {filename} (Page {page})\n\nChunk Text:\n{chunk_text}\n\nExtract chunk metadata:"
            )
        ]).partial(format_instructions=self.chunk_parser.get_format_instructions())
        self.chunk_chain = self.chunk_prompt | self.llm | self.chunk_parser

        self.vector_store: Optional[InMemoryVectorStore] = None
        self.enriched_documents: List[Document] = []

    # -----------------------------------------------------------------
    # Step 1: Document Loading
    # -----------------------------------------------------------------
    def load_data(self, data_dir: str = "Data") -> List[Document]:
        """Loads documents from directory."""
        print(f"\n[Step 1] Loading documents from '{data_dir}'...")
        raw_docs = DocumentLoader.load_directory(data_dir)
        print(f"Total raw pages/documents loaded: {len(raw_docs)}")
        return raw_docs

    # -----------------------------------------------------------------
    # Step 2: Document-Level Metadata Extraction via .abatch()
    # -----------------------------------------------------------------
    async def extract_document_metadata_async(self, raw_docs: List[Document]) -> Dict[str, DocumentOverviewMetadata]:
        """Extracts high-level document metadata for each unique document using .abatch()."""
        print("\n[Step 2] Extracting document-level metadata using ChatNVIDIA.abatch()...")
        
        # Group pages by unique document source
        doc_groups: Dict[str, List[Document]] = {}
        for doc in raw_docs:
            src = doc.metadata.get("source", "unknown")
            doc_groups.setdefault(src, []).append(doc)

        unique_sources = list(doc_groups.keys())
        print(f"Found {len(unique_sources)} unique document(s) to analyze.")

        # Prepare batch inputs
        batch_inputs = []
        for src in unique_sources:
            pages = doc_groups[src]
            sample_content = "\n\n".join([p.page_content for p in pages[:3]])[:3000]
            batch_inputs.append({
                "filename": pages[0].metadata.get("filename", Path(src).name),
                "source": src,
                "content": sample_content
            })

        # Throttle with rate limiter
        await self.rate_limiter.acquire(len(batch_inputs))

        t0 = time.time()
        try:
            results = await self.doc_chain.abatch(batch_inputs)
        except Exception as e:
            print(f"[DocumentMetadata] Batch extraction error: {e}, falling back to default metadata.")
            results = [{
                "document_title": Path(src).stem,
                "document_type": "Document",
                "issuing_authority": "Unknown",
                "effective_date": "Unknown",
                "primary_domain": "General",
                "executive_summary": "Extracted document.",
                "key_stakeholders": []
            } for src in unique_sources]

        doc_meta_map: Dict[str, DocumentOverviewMetadata] = {}
        for src, res in zip(unique_sources, results):
            try:
                parsed = DocumentOverviewMetadata(**res) if isinstance(res, dict) else res
            except Exception:
                parsed = DocumentOverviewMetadata(
                    document_title=Path(src).stem,
                    document_type="Document",
                    issuing_authority="Unknown",
                    effective_date="Unknown",
                    primary_domain="General",
                    executive_summary=str(res),
                    key_stakeholders=[]
                )
            doc_meta_map[src] = parsed
            print(f"  ✓ {parsed.document_title} ({parsed.document_type}) - {parsed.primary_domain}")

        print(f"Document-level metadata extraction completed in {time.time() - t0:.2f}s")
        return doc_meta_map

    # -----------------------------------------------------------------
    # Step 3: Semantic Chunking
    # -----------------------------------------------------------------
    def semantic_chunking(
        self,
        raw_docs: List[Document],
        doc_meta_map: Dict[str, DocumentOverviewMetadata]
    ) -> List[Document]:
        """Splits documents into semantic chunks using NVIDIA Embeddings."""
        print(f"\n[Step 3] Performing Semantic Chunking using NVIDIAEmbeddings ({self.embedding_model_name})...")
        t0 = time.time()

        all_semantic_chunks: List[Document] = []
        for i, doc in enumerate(raw_docs):
            src = doc.metadata.get("source", "unknown")
            doc_meta = doc_meta_map.get(src)

            # Perform semantic chunking on document page/section
            chunks = self.semantic_chunker.split_text(doc.page_content)
            
            for chunk_idx, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) < 30:
                    continue  # skip empty or micro artifacts
                
                chunk_meta = dict(doc.metadata)
                chunk_meta["chunk_index"] = chunk_idx + 1
                if doc_meta:
                    chunk_meta["document_title"] = doc_meta.document_title
                    chunk_meta["document_type"] = doc_meta.document_type
                    chunk_meta["issuing_authority"] = doc_meta.issuing_authority
                    chunk_meta["primary_domain"] = doc_meta.primary_domain
                    chunk_meta["doc_executive_summary"] = doc_meta.executive_summary

                all_semantic_chunks.append(Document(page_content=chunk_text.strip(), metadata=chunk_meta))

        print(f"Semantic chunking produced {len(all_semantic_chunks)} chunks from {len(raw_docs)} pages in {time.time() - t0:.2f}s.")
        return all_semantic_chunks

    # -----------------------------------------------------------------
    # Step 4: Chunk Metadata Enrichment via .abatch() with 40 RPM Throttling
    # -----------------------------------------------------------------
    async def enrich_chunks_async(
        self,
        chunks: List[Document],
        batch_size: Optional[int] = None,
        max_rpm: Optional[int] = None,
        max_retries: int = 3,
    ) -> List[Document]:
        """
        Enriches all semantic chunks in rate-throttled batches via ChatNVIDIA.abatch().
        Guarantees that total requests never exceed `max_rpm` (default: 40 RPM) across any 60-second window.
        """
        actual_batch_size = batch_size or self.batch_size
        actual_max_rpm = max_rpm or self.max_rpm
        
        # Initialize or update rate limiter
        limiter = AsyncRateLimiter(max_rpm=actual_max_rpm)

        print(f"\n[Step 4] Enriching {len(chunks)} chunks with granular metadata via ChatNVIDIA.abatch()...")
        print(f"  • Strict Rate Limit: {actual_max_rpm} RPM (Requests Per Minute)")
        print(f"  • Batch Size: {actual_batch_size} chunks per batch")
        
        # Calculate theoretical minimum duration
        min_total_seconds = (len(chunks) / actual_max_rpm) * 60.0
        print(f"  • Estimated Min Processing Time: {min_total_seconds:.1f}s (~{min_total_seconds / 60:.1f} mins)")
        
        t0 = time.time()

        # Build batch inputs
        batch_inputs = []
        for c in chunks:
            batch_inputs.append({
                "document_title": c.metadata.get("document_title", c.metadata.get("filename", "Document")),
                "primary_domain": c.metadata.get("primary_domain", "Fintech / Regulatory"),
                "filename": c.metadata.get("filename", "unknown"),
                "page": c.metadata.get("page", 1),
                "chunk_text": c.page_content
            })

        enriched_results: List[Any] = []
        total_batches = (len(batch_inputs) + actual_batch_size - 1) // actual_batch_size
        
        for b_idx in range(total_batches):
            start = b_idx * actual_batch_size
            end = min(start + actual_batch_size, len(batch_inputs))
            sub_batch = batch_inputs[start:end]
            sub_batch_len = len(sub_batch)

            # 1. Acquire quota from rate limiter (blocks until rate window permits)
            waited = await limiter.acquire(sub_batch_len)
            current_rpm = limiter.current_rpm_usage

            elapsed = time.time() - t0
            progress_pct = (start / len(batch_inputs)) * 100
            print(f"  -> Batch {b_idx + 1}/{total_batches} ({sub_batch_len} chunks) | Window RPM: {current_rpm}/{actual_max_rpm} | Progress: {progress_pct:.1f}%")
            if waited > 0.5:
                print(f"     [RPM Pacing] Paused {waited:.1f}s to respect 40 RPM limit.")

            # 2. Execute .abatch() with exponential backoff retry on 429 / rate limits
            sub_results = None
            for attempt in range(1, max_retries + 1):
                try:
                    sub_results = await self.chunk_chain.abatch(sub_batch)
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    is_rate_limit = "429" in err_str or "rate limit" in err_str or "too many requests" in err_str
                    if is_rate_limit or attempt < max_retries:
                        backoff = 6.0 * (2 ** (attempt - 1))
                        print(f"     [Retry {attempt}/{max_retries}] Caught error ({e}). Backing off {backoff:.1f}s...")
                        await asyncio.sleep(backoff)
                    else:
                        print(f"     [Failed] Batch {b_idx + 1} failed after {max_retries} attempts: {e}")
                        break

            if sub_results and isinstance(sub_results, list) and len(sub_results) == sub_batch_len:
                enriched_results.extend(sub_results)
            else:
                # Apply fallback metadata for failed sub-batch
                for item in sub_batch:
                    fallback = {
                        "chunk_title": item["chunk_text"][:50].strip() + "...",
                        "chunk_summary": item["chunk_text"][:150].strip(),
                        "entities": [],
                        "topics": [item["primary_domain"]],
                        "compliance_mandates": [],
                        "potential_questions": []
                    }
                    enriched_results.append(fallback)

        # Merge enriched metadata into chunks
        enriched_docs: List[Document] = []
        for chunk, enriched_meta in zip(chunks, enriched_results):
            final_meta = dict(chunk.metadata)
            if isinstance(enriched_meta, dict):
                final_meta["chunk_title"] = enriched_meta.get("chunk_title", "")
                final_meta["chunk_summary"] = enriched_meta.get("chunk_summary", "")
                final_meta["entities"] = enriched_meta.get("entities", [])
                final_meta["topics"] = enriched_meta.get("topics", [])
                final_meta["compliance_mandates"] = enriched_meta.get("compliance_mandates", [])
                final_meta["potential_questions"] = enriched_meta.get("potential_questions", [])
            
            enriched_docs.append(Document(page_content=chunk.page_content, metadata=final_meta))

        self.enriched_documents = enriched_docs
        total_time = time.time() - t0
        avg_rpm = (len(chunks) / max(total_time, 1)) * 60
        print(f"\nChunk metadata enrichment completed in {total_time:.2f}s (Effective Rate: {avg_rpm:.1f} RPM).")
        return enriched_docs

    # -----------------------------------------------------------------
    # Step 5: Vector Store Indexing & Storage
    # -----------------------------------------------------------------
    def build_vector_store(self, enriched_docs: Optional[List[Document]] = None) -> InMemoryVectorStore:
        """Indexes enriched chunks into vector store."""
        docs_to_index = enriched_docs or self.enriched_documents
        print(f"\n[Step 5] Indexing {len(docs_to_index)} enriched chunks into Vector Store using {self.embedding_model_name}...")
        t0 = time.time()
        
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.vector_store.add_documents(docs_to_index)
        print(f"Successfully indexed {len(docs_to_index)} chunks in {time.time() - t0:.2f}s.")
        return self.vector_store

    def export_processed_data(
        self,
        json_path: str = "Data/processed_documents.json",
        jsonl_path: str = "Data/processed_documents.jsonl",
        append: bool = True
    ):
        """Exports enriched chunks and metadata to JSON/JSONL format, optionally appending to existing files."""
        if not self.enriched_documents:
            print("[Export] No enriched documents to export.")
            return

        export_data = []
        
        # Determine starting index for new chunks
        start_idx = 1
        existing_data = []
        
        if append and os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    if existing_data:
                        # Extract the maximum chunk index to continue incrementing
                        max_idx = 0
                        for item in existing_data:
                            if item.get("id", "").startswith("chunk_"):
                                try:
                                    idx = int(item["id"].split("_")[1])
                                    max_idx = max(max_idx, idx)
                                except ValueError:
                                    pass
                        start_idx = max_idx + 1
            except Exception as e:
                print(f"[Export] Could not load existing JSON for appending: {e}")

        for idx, doc in enumerate(self.enriched_documents, start=start_idx):
            export_data.append({
                "id": f"chunk_{idx}",
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        # Save JSON
        os.makedirs(Path(json_path).parent, exist_ok=True)
        final_data = existing_data + export_data if append else export_data
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        print(f"[Export] Saved {len(export_data)} new chunks (Total: {len(final_data)}) to '{json_path}'")

        # Save JSONL
        mode = "a" if append and os.path.exists(jsonl_path) else "w"
        with open(jsonl_path, mode, encoding="utf-8") as f:
            for item in export_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[Export] Saved {len(export_data)} enriched chunks to '{json_path}'")

    # -----------------------------------------------------------------
    # Query & Retrieval Helper
    # -----------------------------------------------------------------
    def search(self, query: str, k: int = 4, filter_dict: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Performs semantic similarity search over indexed chunks."""
        if not self.vector_store:
            raise ValueError("Vector store has not been initialized. Run pipeline first.")
        return self.vector_store.similarity_search(query, k=k)

    # -----------------------------------------------------------------
    # Complete Pipeline Execution
    # -----------------------------------------------------------------
    async def run_pipeline_async(self, data_dir: str = "Data") -> List[Document]:
        """Executes the entire document loading, metadata extraction, chunking, enrichment and indexing pipeline."""
        start_time = time.time()
        print("=" * 80)
        print("STARTING NVIDIA-POWERED DOCUMENT ENRICHMENT PIPELINE")
        print(f"LLM: {self.llm_model_name} | Embeddings: {self.embedding_model_name} | Rate Limit: {self.max_rpm} RPM")
        print("=" * 80)

        # 1. Load documents
        raw_docs = self.load_data(data_dir)
        if not raw_docs:
            print("[Pipeline] No documents found to process.")
            return []

        # 2. Document-level metadata extraction (.abatch)
        doc_meta_map = await self.extract_document_metadata_async(raw_docs)

        # 3. Semantic chunking
        semantic_chunks = self.semantic_chunking(raw_docs, doc_meta_map)

        # 4. Chunk-level metadata enrichment (.abatch with 40 RPM limit)
        enriched_chunks = await self.enrich_chunks_async(semantic_chunks)

        # 5. Vector Store indexing
        self.build_vector_store(enriched_chunks)

        # 6. Save results to disk
        self.export_processed_data()

        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
        print(f"Total Enriched Chunks: {len(enriched_chunks)}")
        print("=" * 80)
        return enriched_chunks


# Synchronous convenience wrapper
def run_pipeline(data_dir: str = "Data", **kwargs) -> DocumentPipeline:
    pipeline = DocumentPipeline(**kwargs)
    asyncio.run(pipeline.run_pipeline_async(data_dir=data_dir))
    return pipeline


if __name__ == "__main__":
    # Run pipeline directly
    pipeline = run_pipeline(data_dir="Data")
    
    # Test query
    test_query = "What are the rules regarding loan disbursals and fees paid to LSPs?"
    print(f"\n[Test Query] '{test_query}'\n")
    results = pipeline.search(test_query, k=2)
    for i, res in enumerate(results, 1):
        print(f"--- Search Result {i} ---")
        print(f"Title: {res.metadata.get('chunk_title')}")
        print(f"Summary: {res.metadata.get('chunk_summary')}")
        print(f"Entities: {res.metadata.get('entities')}")
        print(f"Compliance Mandates: {res.metadata.get('compliance_mandates')}")
        print(f"Content: {res.page_content[:200]}...")
