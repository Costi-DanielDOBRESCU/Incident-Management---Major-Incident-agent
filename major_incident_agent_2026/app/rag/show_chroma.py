from app.rag.chroma_client import ALL_COLLECTIONS, get_or_create_collection

for coll_name in ALL_COLLECTIONS:
    coll = get_or_create_collection(coll_name)
    print(f"\n=================== COLECȚIA: {coll_name} ({coll.count()} documente) ===================")
    
    # Preia toate datele salvate în colecție
    data = coll.get(include=["documents", "metadatas", "embeddings"])
    
    for doc_id, doc_text, metadata in zip(data["ids"], data["documents"], data["metadatas"]):
        print(f"\n[ID]: {doc_id}")
        print(f"[Summary]: {metadata.get('summary')}")
        print(f"[Service]: {metadata.get('service')}")
        print(f"[Content]: {doc_text[:120]}...") # Primele 120 de caractere