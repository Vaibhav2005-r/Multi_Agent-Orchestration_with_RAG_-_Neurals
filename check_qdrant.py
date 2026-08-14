"""
Inspect Qdrant Database Collection
"""

import sys
from db_client import get_qdrant_client, DEFAULT_COLLECTION

def inspect_qdrant():
    client = get_qdrant_client()
    collection_name = DEFAULT_COLLECTION

    if not client.collection_exists(collection_name):
        print(f"⚠️  Collection '{collection_name}' not found in Qdrant database.")
        print("ℹ️  To index your documents, please run:")
        print("     python qdrant_indexer.py")
        print("   or run the full ingestion pipeline:")
        print("     python ingestion_pipeline.py\n")
        return

    # Get collection info
    info = client.get_collection(collection_name)
    print(f"✅ Collection Found: {collection_name}")
    print(f"   Total Points: {info.points_count}")
    print(f"   Vector Dimensions: {info.config.params.vectors.size}")
    print(f"   Distance Metric: {info.config.params.vectors.distance.name}")

    # Fetch 1 sample point
    print("\n--- Sample Point Metadata ---")
    res = client.scroll(collection_name=collection_name, limit=1)
    if res and res[0]:
        point = res[0][0]
        print(f"   Vector size: {len(point.vector) if point.vector else 'Not fetched by default'}")
        print(f"   Payload Keys: {list(point.payload.keys())}")
        
        for k, v in point.payload.items():
            if k in ['page_content']: 
                continue
            print(f"     {k}: {v}")
    else:
        print("   (Collection is empty)")

if __name__ == "__main__":
    inspect_qdrant()
