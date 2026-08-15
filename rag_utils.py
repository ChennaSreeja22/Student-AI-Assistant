import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text_with_metadata(
    text,
    pdf_name,
    chunk_size=450,
    overlap=75
):
    words = text.split()

    chunks = []
    metadata = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):

        chunk_words = words[i:i + chunk_size]

        if not chunk_words:
            continue

        chunk = " ".join(chunk_words)

        chunks.append(chunk)

        metadata.append({
            "text": chunk,
            "file": pdf_name,
            "page": None
        })

        if i + chunk_size >= len(words):
            break

    return chunks, metadata

def store_chunks(chunks):

    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index, chunks


def retrieve_chunks_with_metadata(
    query,
    index,
    metadata,
    n_results=5,
    similarity_threshold=0.25
):

    query_embedding = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    n_results = min(n_results, index.ntotal)

    if n_results == 0:
        return []

    similarities, indices = index.search(
        query_embedding,
        n_results
    )

    results = []

    for similarity, idx in zip(
        similarities[0],
        indices[0]
    ):

        if idx < 0:
            continue

        if similarity < similarity_threshold:
            continue

        result = metadata[idx].copy()

        result["similarity"] = float(similarity)

        results.append(result)

    return results


def retrieve_chunks(
    query,
    index,
    chunks,
    n_results=5,
    similarity_threshold=0.25
):

    query_embedding = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    n_results = min(n_results, index.ntotal)

    if n_results == 0:
        return []

    similarities, indices = index.search(
        query_embedding,
        n_results
    )

    results = []

    for similarity, idx in zip(
        similarities[0],
        indices[0]
    ):

        if idx < 0:
            continue

        if similarity < similarity_threshold:
            continue

        results.append(chunks[idx])

    return results