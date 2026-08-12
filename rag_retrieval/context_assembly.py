import os
import time
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document

# Deduplication
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.document_transformers import EmbeddingsRedundantFilter

# Reranking
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

# Context Packaging
from llama_index.core.postprocessor import LongContextReorder
from llama_index.core.schema import TextNode, NodeWithScore

load_dotenv()

class SemanticDeduplicator:
    def __init__(self, embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2", threshold: float = 0.95):
        self.api_key = os.environ.get("NVIDIA_API_KEY")
        self.embeddings = NVIDIAEmbeddings(model=embedding_model, api_key=self.api_key, truncate="END")
        self.filter = EmbeddingsRedundantFilter(
            embeddings=self.embeddings,
            similarity_threshold=threshold
        )

    @staticmethod
    def _hash_dedup(docs: List[Document]) -> List[Document]:
        """Fast exact-text deduplication using content hash before embedding API call."""
        seen, unique = set(), []
        for doc in docs:
            key = " ".join(doc.page_content.split())  # normalise whitespace
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique

    def deduplicate(self, docs: List[Document]) -> List[Document]:
        """Removes duplicate chunks: exact-match first, then semantic similarity."""
        # 1. Fast hash dedup (zero API cost)
        docs = self._hash_dedup(docs)
        if len(docs) <= 1:
            return docs
        # 2. Semantic similarity dedup via NVIDIA Embeddings
        return self.filter.transform_documents(docs)


class RelevanceReranker:
    def __init__(self, top_n: int = 5):
        self.reranker = FlashrankRerank(top_n=top_n)
        
    def rerank(self, docs: List[Document], query: str) -> List[Document]:
        """Reranks chunks using FlashRank locally."""
        return self.reranker.compress_documents(docs, query=query)


class ContextPackager:
    def __init__(self):
        self.reorder = LongContextReorder()
        
    def package(self, docs: List[Document]) -> List[Document]:
        """Reorders documents to combat 'lost in the middle' effect."""
        nodes = []
        for doc in docs:
            node = TextNode(text=doc.page_content, metadata=doc.metadata)
            score = doc.metadata.get("relevance_score", 0.0)
            nodes.append(NodeWithScore(node=node, score=score))
            
        reordered_nodes = self.reorder.postprocess_nodes(nodes)
        
        reordered_docs = []
        for n in reordered_nodes:
            reordered_docs.append(Document(page_content=n.node.text, metadata=n.node.metadata))
            
        return reordered_docs


class AdvancedRAGPipeline:
    def __init__(self, top_n_results: int = 5):
        print("Initializing Advanced RAG Pipeline Components...")
        self.deduplicator = SemanticDeduplicator()
        self.reranker = RelevanceReranker(top_n=top_n_results)
        self.packager = ContextPackager()
        
    def process(self, query: str, raw_docs: List[Document]) -> List[Document]:
        print(f"\n[Advanced RAG] Starting context assembly for query: '{query}'")
        print(f"  -> Initial Documents: {len(raw_docs)}")
        
        # Step A: Deduplicate
        t0 = time.time()
        unique_docs = self.deduplicator.deduplicate(raw_docs)
        print(f"  -> Semantic Deduplication: {len(raw_docs)} -> {len(unique_docs)} docs (took {time.time()-t0:.2f}s)")
        
        if not unique_docs:
            return []
            
        # Step B: Rerank
        t1 = time.time()
        reranked_docs = self.reranker.rerank(docs=unique_docs, query=query)
        print(f"  -> FlashRank Reranking: Top {len(reranked_docs)} docs (took {time.time()-t1:.2f}s)")
        
        # Step C: Context Package (LongContextReorder)
        t2 = time.time()
        final_docs = self.packager.package(reranked_docs)
        print(f"  -> Long-Context Reordering applied (took {time.time()-t2:.2f}s)")
        
        return final_docs

if __name__ == "__main__":
    # Mock Test
    test_docs = [
        Document(page_content="The LSP must not charge the borrower any fees directly.", metadata={"source": "doc1.txt"}),
        Document(page_content="LSPs are strictly prohibited from charging direct fees to the borrower.", metadata={"source": "doc2.txt"}), # Semantic duplicate
        Document(page_content="Lending interest rates depend on the market benchmark.", metadata={"source": "doc3.txt"}),
        Document(page_content="The penalty for delayed payments is 2% per month.", metadata={"source": "doc4.txt"}),
        Document(page_content="Digital lending platforms must display the APR clearly.", metadata={"source": "doc5.txt"}),
    ]
    
    pipeline = AdvancedRAGPipeline(top_n_results=3)
    query = "What are the rules regarding fees paid to LSPs?"
    
    results = pipeline.process(query, test_docs)
    
    print("\n--- FINAL ASSEMBLED CONTEXT ---")
    for i, res in enumerate(results, 1):
        score = res.metadata.get("relevance_score", "N/A")
        print(f"\nResult {i} (Score: {score}):\n{res.page_content}")
