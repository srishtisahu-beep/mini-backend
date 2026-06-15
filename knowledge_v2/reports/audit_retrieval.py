from sentence_transformers import SentenceTransformer
import chromadb

# ==================================================
# CONFIG
# ==================================================

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

# ==================================================
# LOAD QUERIES
# ==================================================

with open(
    "insurance_test_queries.txt",
    "r",
    encoding="utf-8"
) as f:

    queries = [
        line.strip()
        for line in f
        if line.strip()
    ]

# ==================================================
# TEST
# ==================================================

for query in queries:

    print("\n")
    print("=" * 120)
    print("QUERY:", query)
    print("=" * 120)

    emb = model.encode(
        query
    ).tolist()

    results = collection.query(
        query_embeddings=[emb],
        n_results=3
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    for rank, (doc, meta) in enumerate(
        zip(docs, metas),
        start=1
    ):

        print("\n")
        print("-" * 100)
        print(f"TOP RESULT {rank}")
        print("-" * 100)

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
            "CATEGORY:",
            meta.get(
                "category",
                "N/A"
            )
        )

        print("\nCONTENT:\n")

        print(
            doc[:1000]
        )

        print("\n")