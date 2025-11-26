# src/build_index.py
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_DIR = Path("data")
INDEX_DIR = Path("index/chroma")
COLLECTION_NAME = "nfl_knowledge"


def get_chroma_client():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(INDEX_DIR), settings=Settings(anonymized_telemetry=False))


def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [e.embedding for e in resp.data]


def load_documents() -> list[tuple[str, str]]:
    docs = []
    for txt_path in DATA_DIR.rglob("*.txt"):
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        docs.append((str(txt_path), text))
    return docs


def chunk_text(text: str, max_chars: int = 800) -> list[str]:
    # simplistic chunker on paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 2 <= max_chars:
            current = current + ("\n\n" if current else "") + p
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks


def main():
    chroma_client = get_chroma_client()

    # Drop and recreate for a clean rebuild
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(COLLECTION_NAME)

    docs = load_documents()
    print(f"Found {len(docs)} documents under data/")

    doc_ids = []
    contents = []
    metadatas = []

    for path_str, text in docs:
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            doc_id = f"{path_str}:{i}"
            doc_ids.append(doc_id)
            contents.append(chunk)
            metadatas.append({"source": path_str})

    print(f"Chunked into {len(contents)} pieces. Embedding and indexing...")

    # Embedding in batches
    batch_size = 32
    for start in tqdm(range(0, len(contents), batch_size)):
        end = start + batch_size
        batch_texts = contents[start:end]
        batch_ids = doc_ids[start:end]
        batch_meta = metadatas[start:end]

        vectors = embed_texts(batch_texts)
        collection.add(
            ids=batch_ids,
            embeddings=vectors,
            documents=batch_texts,
            metadatas=batch_meta,
        )

    print("Index build complete.")


if __name__ == "__main__":
    main()
