"""
Shared Database Client and Path Configuration
=============================================
Provides cross-platform absolute path resolution and a shared QdrantClient
singleton to prevent file lock contention and path resolution errors on Windows and macOS.
"""

import os
import sys
import atexit
from pathlib import Path
from typing import Optional
from qdrant_client import QdrantClient

# Auto-reconfigure stdout/stderr encoding on Windows to prevent charmap errors
if sys.platform == "win32":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Absolute base project directory
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
DEFAULT_QDRANT_PATH = str(DATA_DIR / "qdrant_db")
DEFAULT_JSON_PATH = str(DATA_DIR / "processed_documents.json")
DEFAULT_COLLECTION = "fintech_documents"

_shared_qdrant_client: Optional[QdrantClient] = None

def _cleanup_client():
    global _shared_qdrant_client
    if _shared_qdrant_client is not None:
        try:
            _shared_qdrant_client.close()
        except Exception:
            pass
        _shared_qdrant_client = None

atexit.register(_cleanup_client)

def get_qdrant_client(qdrant_path: Optional[str] = None) -> QdrantClient:
    """
    Returns a shared or dedicated QdrantClient with normalized absolute path.
    """
    global _shared_qdrant_client
    target_path = os.path.abspath(qdrant_path) if qdrant_path else DEFAULT_QDRANT_PATH
    
    # Ensure directory exists
    os.makedirs(target_path, exist_ok=True)
    
    if target_path == DEFAULT_QDRANT_PATH:
        if _shared_qdrant_client is None:
            _shared_qdrant_client = QdrantClient(path=target_path)
        return _shared_qdrant_client
    
    return QdrantClient(path=target_path)
