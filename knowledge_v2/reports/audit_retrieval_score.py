from sentence_transformers import SentenceTransformer
import chromadb

DB_PATH = "./chroma_db_v2"
COLLECTION_NAME = "insurance_kb_v2"

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

client = chromadb.PersistentClient(
    path=DB_PATH
)

collection = client.get_collection(
    COLLECTION_NAME
)

with open(
    "insurance_test_queries.txt",
    "r",
    encoding="utf-8"
) as f:

    queries = [
        q.strip()
        for q in f
        if q.strip()
    ]

for query in queries:

    emb = model.encode(
        query
    ).tolist()

    result = collection.query(
        query_embeddings=[emb],
        n_results=1
    )

    doc = result["documents"][0][0]
    meta = result["metadatas"][0][0]

    print("=" * 100)

    print(
        "QUERY:",
        query
    )

    print(
        "PRODUCT:",
        meta.get(
            "product_name",
            "N/A"
        )
    )

    print(
        "TYPE:",
        meta.get(
            "chunk_type",
            "N/A"
        )
    )

    print(
        "SOURCE:",
        meta.get(
            "source_file",
            "N/A"
        )
    )

    print(
        "PREVIEW:",
        doc[:250].replace(
            "\n",
            " "
        )
    )

    print()