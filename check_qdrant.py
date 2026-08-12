from qdrant_client import QdrantClient

# Connect to the local DB
client = QdrantClient(path="Data/qdrant_db_optimized")
collection_name = "fintech_documents_optimized"

# Get collection info
info = client.get_collection(collection_name)
print(f"Collection Name: {collection_name}")
print(f"Total Points: {info.points_count}")
print(f"Vector Dimensions: {info.config.params.vectors.size}")
print(f"Distance Metric: {info.config.params.vectors.distance.name}")

# Fetch 1 sample point
print("\n--- Sample Point Metadata ---")
res = client.scroll(collection_name=collection_name, limit=1)
if res[0]:
    point = res[0][0]
    print(f"Vector size explicitly returned: {len(point.vector) if point.vector else 'Not fetched by default'}")
    print(f"Payload Keys: {list(point.payload.keys())}")
    
    # Print some payload details (exclude the heavy ones if necessary)
    for k, v in point.payload.items():
        if k in ['page_content']: 
            continue
        print(f"  {k}: {v}")
