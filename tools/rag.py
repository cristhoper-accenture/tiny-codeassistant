"""
RAG (Retrieval-Augmented Generation) tools.

Storage layout per collection:
  rag_store/<collection>/
      meta.json        — {doc_id: {source, chunks: [{id, text}]}}
      embeddings.npy   — float32 matrix (n_chunks × embed_dim)
      chunk_ids.json   — ordered list of chunk IDs matching matrix rows
"""

import os
import json
import hashlib
import math
import urllib.request
import html.parser
import numpy as np

import llm
from config import RAG_DIR, EMBED_MODEL, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP


# ── Helpers ────────────────────────────────────────────────────────────────────

def _collection_dir(collection: str) -> str:
    d = os.path.join(RAG_DIR, collection)
    os.makedirs(d, exist_ok=True)
    return d


def _meta_path(collection: str) -> str:
    return os.path.join(_collection_dir(collection), "meta.json")


def _emb_path(collection: str) -> str:
    return os.path.join(_collection_dir(collection), "embeddings.npy")


def _ids_path(collection: str) -> str:
    return os.path.join(_collection_dir(collection), "chunk_ids.json")


def _load_meta(collection: str) -> dict:
    p = _meta_path(collection)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def _save_meta(collection: str, meta: dict) -> None:
    with open(_meta_path(collection), "w") as f:
        json.dump(meta, f, indent=2)


def _load_store(collection: str) -> tuple[np.ndarray | None, list[str]]:
    """Return (embeddings_matrix, chunk_ids_list). Matrix is None if empty."""
    ep, ip = _emb_path(collection), _ids_path(collection)
    if not os.path.exists(ep) or not os.path.exists(ip):
        return None, []
    with open(ip) as f:
        ids = json.load(f)
    if not ids:
        return None, []
    return np.load(ep), ids


def _save_store(collection: str, matrix: np.ndarray, ids: list[str]) -> None:
    np.save(_emb_path(collection), matrix)
    with open(_ids_path(collection), "w") as f:
        json.dump(ids, f)


# ── Chunking ───────────────────────────────────────────────────────────────────

def _chunk(text: str) -> list[str]:
    """Split on word boundaries with overlap."""
    words = text.split()
    chunks, buf, buf_len = [], [], 0
    for word in words:
        buf.append(word)
        buf_len += len(word) + 1
        if buf_len >= RAG_CHUNK_SIZE:
            chunks.append(" ".join(buf))
            # keep last N chars worth of words as overlap
            overlap_words = []
            overlap_len = 0
            for w in reversed(buf):
                overlap_len += len(w) + 1
                overlap_words.insert(0, w)
                if overlap_len >= RAG_CHUNK_OVERLAP:
                    break
            buf = overlap_words
            buf_len = overlap_len
    if buf:
        chunks.append(" ".join(buf))
    return [c for c in chunks if c.strip()]


def _doc_id(source: str) -> str:
    return hashlib.sha1(source.encode()).hexdigest()[:12]


def _chunk_id(doc_id: str, i: int) -> str:
    return f"{doc_id}_{i}"


# ── URL text extraction ────────────────────────────────────────────────────────

class _TextExtractor(html.parser.HTMLParser):
    _SKIP = {"script", "style", "head", "nav", "footer", "aside"}

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth:
            stripped = data.strip()
            if stripped:
                self.parts.append(stripped)


def _fetch_url(url: str) -> str:
    from config import RAG_FETCH_TIMEOUT
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=RAG_FETCH_TIMEOUT) as resp:
        raw = resp.read().decode(resp.headers.get_content_charset("utf-8"), errors="replace")
    parser = _TextExtractor()
    parser.feed(raw)
    return " ".join(parser.parts)


# ── Core ingest ────────────────────────────────────────────────────────────────

def _ingest(source: str, text: str, collection: str) -> str:
    """Chunk, embed, and store text. Returns summary string."""
    text = text.strip()
    if not text:
        return "ERROR: empty document"

    doc_id = _doc_id(source)
    chunks = _chunk(text)
    if not chunks:
        return "ERROR: no chunks produced"

    # Embed all chunks
    embeddings = []
    for c in chunks:
        embeddings.append(llm.embed(c, model=EMBED_MODEL))

    new_matrix = np.array(embeddings, dtype=np.float32)
    new_ids = [_chunk_id(doc_id, i) for i in range(len(chunks))]

    # Load existing store and remove old version of this doc if present
    old_matrix, old_ids = _load_store(collection)
    meta = _load_meta(collection)

    if doc_id in meta:
        old_chunk_ids = {c["id"] for c in meta[doc_id]["chunks"]}
        keep = [i for i, cid in enumerate(old_ids) if cid not in old_chunk_ids]
        if keep and old_matrix is not None:
            old_matrix = old_matrix[keep]
            old_ids = [old_ids[i] for i in keep]
        else:
            old_matrix, old_ids = None, []

    # Merge
    if old_matrix is not None and len(old_ids) > 0:
        combined_matrix = np.vstack([old_matrix, new_matrix])
        combined_ids = old_ids + new_ids
    else:
        combined_matrix = new_matrix
        combined_ids = new_ids

    _save_store(collection, combined_matrix, combined_ids)

    meta[doc_id] = {
        "source": source,
        "chunks": [{"id": _chunk_id(doc_id, i), "text": c} for i, c in enumerate(chunks)],
    }
    _save_meta(collection, meta)

    return f"Ingested '{source}' into collection '{collection}': {len(chunks)} chunks, {len(text)} chars."


# ── Public API ─────────────────────────────────────────────────────────────────

def add_text(name: str, text: str, collection: str = "default") -> str:
    return _ingest(name, text, collection)


def add_file(path: str, collection: str = "default", cwd: str = ".") -> str:
    if not os.path.isabs(path):
        path = os.path.join(cwd, path)
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return f"ERROR: file not found: {path}"
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return _ingest(path, text, collection)


def add_url(url: str, collection: str = "default") -> str:
    try:
        text = _fetch_url(url)
    except Exception as e:
        return f"ERROR fetching {url}: {e}"
    return _ingest(url, text, collection)


def search(query: str, collection: str = "default", top_k: int = 5) -> str:
    matrix, ids = _load_store(collection)
    if matrix is None:
        return f"Collection '{collection}' is empty."

    meta = _load_meta(collection)
    # Build chunk_id → text lookup
    chunk_text: dict[str, str] = {}
    chunk_source: dict[str, str] = {}
    for doc in meta.values():
        for c in doc["chunks"]:
            chunk_text[c["id"]] = c["text"]
            chunk_source[c["id"]] = doc["source"]

    q_vec = np.array(llm.embed(query, model=EMBED_MODEL), dtype=np.float32)
    # Cosine similarity: normalize then dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normed = matrix / np.where(norms > 0, norms, 1)
    q_norm = q_vec / (np.linalg.norm(q_vec) or 1)
    scores = normed @ q_norm

    top_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for i in top_idx:
        cid = ids[i]
        score = float(scores[i])
        results.append(
            f"[score={score:.3f}] ({chunk_source.get(cid, '?')})\n{chunk_text.get(cid, '')}"
        )
    return "\n\n---\n\n".join(results)


def list_docs(collection: str = "default") -> str:
    meta = _load_meta(collection)
    if not meta:
        return f"Collection '{collection}' is empty."
    lines = [f"Collection: {collection}"]
    for doc_id, info in meta.items():
        n = len(info["chunks"])
        lines.append(f"  • [{doc_id}] {info['source']}  ({n} chunks)")
    return "\n".join(lines)


def list_collections() -> str:
    if not os.path.isdir(RAG_DIR):
        return "No RAG collections yet."
    cols = [d for d in os.listdir(RAG_DIR) if os.path.isdir(os.path.join(RAG_DIR, d))]
    if not cols:
        return "No RAG collections yet."
    return "Collections: " + ", ".join(cols)


def delete_doc(source: str, collection: str = "default") -> str:
    meta = _load_meta(collection)
    doc_id = _doc_id(source)
    # also accept direct doc_id
    if doc_id not in meta:
        doc_id = source if source in meta else None
    if doc_id is None:
        return f"Document '{source}' not found in collection '{collection}'."

    old_chunk_ids = {c["id"] for c in meta[doc_id]["chunks"]}
    matrix, ids = _load_store(collection)
    if matrix is not None:
        keep = [i for i, cid in enumerate(ids) if cid not in old_chunk_ids]
        if keep:
            _save_store(collection, matrix[keep], [ids[i] for i in keep])
        else:
            _save_store(collection, np.empty((0,), dtype=np.float32), [])

    src = meta[doc_id]["source"]
    del meta[doc_id]
    _save_meta(collection, meta)
    return f"Deleted '{src}' from collection '{collection}'."
