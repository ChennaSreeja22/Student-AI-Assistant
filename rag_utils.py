import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text_with_metadata(text, pdf_name, chunk_size=500):
    words = text.split()
    chunks = []
    metadata = []
    total_words = len(words)
    for i in range(0, total_words, chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        page_estimate = int((i / total_words) * 100) + 1
        chunks.append(chunk)
        metadata.append({
            "text": chunk,
            "file": pdf_name,
            "page": page_estimate
        })
    return chunks, metadata

def store_chunks(chunks):
    embeddings = embedder.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

def retrieve_chunks_with_metadata(query, index, metadata, n_results=3):
    query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    distances, indices = index.search(query_embedding, n_results)
    results = [metadata[i] for i in indices[0]]
    return results

def retrieve_chunks(query, index, chunks, n_results=3):
    query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    distances, indices = index.search(query_embedding, n_results)
    return [chunks[i] for i in indices[0]]