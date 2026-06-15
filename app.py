from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import chromadb
from sentence_transformers import SentenceTransformer

# ============================================================

# CONFIG

# ============================================================

CHROMA_PATH = "./kb_chroma"
COLLECTION_NAME = "bajaj_kb"

SYSTEM_PROMPT_FILE = "system_prompt.txt"
SKILLS_FILE = "skills_master.txt"

# ============================================================

# HELPERS

# ============================================================

def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

# Load prompts now so tomorrow you only uncomment LLM section

SYSTEM_PROMPT = read_file(SYSTEM_PROMPT_FILE)
SKILLS = read_file(SKILLS_FILE)

MASTER_SYSTEM_PROMPT = f"""
{SYSTEM_PROMPT}

========================================================

{SKILLS}
"""

# ============================================================

# APP

# ============================================================

app = FastAPI(
    title="Mini Insurance Backend",
    version="1.0"
)

# ============================================================

# CORS

# ============================================================

app.add_middleware(
CORSMiddleware,
allow_origins=["*"],   # tighten later
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

# ============================================================

# LOAD EMBEDDING MODEL

# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
"BAAI/bge-base-en-v1.5"
)

print("Embedding model loaded.")

# ============================================================

# LOAD CHROMA

# ============================================================

print("Loading ChromaDB...")

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

print("\nCollections Found:")

for c in client.list_collections():
    print(c.name)


collection = client.get_collection(
    name=COLLECTION_NAME
)

print("ChromaDB loaded.")
print("Documents:", collection.count())

# ============================================================

# REQUEST MODEL

# ============================================================

class ChatRequest(BaseModel):
    message: str

# ============================================================

# ROUTES

# ============================================================

@app.get("/")
def root():


    return {
        "status": "running",
        "service": "Mini Backend"
    }


@app.get("/health")
def health():


    return {
        "status": "healthy",
        "collection": COLLECTION_NAME,
        "documents": collection.count()
    }

@app.get("/debug")
def debug():

    query_embedding = embedding_model.encode(
        "Care Supreme",
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20
    )

    return {
        "files": [
            m.get("source_file")
            for m in results["metadatas"][0]
        ]
    }


@app.post("/chat")
def chat(req: ChatRequest):

    query = req.message.strip()

    print("\n" + "=" * 100)
    print("QUERY:", query)
    print("=" * 100)

    # ========================================================
    # EMBEDDING
    # ========================================================

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # ========================================================
    # RETRIEVAL
    # ========================================================

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20,
        include=["documents", "metadatas", "distances"]
    )

    distances = results["distances"][0]

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    print("\nDistances:")
    print(distances)

    filtered_docs = []
    filtered_metas = []

    for doc, meta, dist in zip(docs, metas, distances):

        if dist < 0.8:
            filtered_docs.append(doc)
            filtered_metas.append(meta)

    docs = filtered_docs
    metas = filtered_metas

    if not docs:
        return {
            "answer": "No relevant information found.",
            "sources": [],
            "retrieved_chunks": []
        }

    retrieved_chunks = []

    for doc, meta in zip(docs, metas):

        retrieved_chunks.append({
            "content": doc,
            "source_file": (
                meta.get("source_file")
                if meta else "unknown"
            ),
            "section": (
                meta.get("section")
                if meta else "unknown"
            )
        })

    print("\nTOP RETRIEVED CHUNKS:\n")

    for idx, chunk in enumerate(retrieved_chunks):

        print(f"\n--- CHUNK {idx+1} ---")
        print(chunk["content"][:800])

    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context = "\n\n".join(
        chunk["content"]
        for chunk in retrieved_chunks
    )

    # ========================================================
    # LLM SECTION
    # ========================================================
    #
    # Uncomment tomorrow
    #

    # from openai import OpenAI
    #
    # llm_client = OpenAI(
    #     base_url="http://13.127.232.184:8000/v1",
    #     api_key="dummy"
    # )
    #
    # user_prompt = f'''
    # Knowledge Base Context:
    #
    # {context}
    #
    # Customer Query:
    #
    # {query}
    # '''
    #
    # response = llm_client.chat.completions.create(
    #     model="mini",
    #     temperature=0.1,
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": MASTER_SYSTEM_PROMPT
    #         },
    #         {
    #             "role": "user",
    #             "content": user_prompt
    #         }
    #     ]
    # )
    #
    # answer = response.choices[0].message.content
    #
    # return {
    #     "answer": answer,
    #     "sources": list(
    #         set(
    #             chunk["source_file"]
    #             for chunk in retrieved_chunks
    #         )
    #     )
    # }

    # ========================================================
    # RETRIEVAL ONLY MODE
    # ========================================================

    return {
        "mode": "retrieval_only",
        "query": query,
        "sources": [
            {
                "file": chunk["source_file"],
                "section": chunk["section"]
            }
            for chunk in retrieved_chunks
        ],
        "retrieved_chunks": retrieved_chunks
    }

