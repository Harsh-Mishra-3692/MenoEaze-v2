# ingest_clinical_books.py — FINAL ELITE (SEMANTIC + ATOMIC + RESUMABLE)

import uuid
import logging
import hashlib
import time
import re
from pathlib import Path
from typing import List, Dict, Set

import numpy as np
from tqdm import tqdm
from pypdf import PdfReader

from ml_engine.hf_embedder import HFEmbedder
from ml_engine.db_client import insert_bulk, fetch_table

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
BOOK_DIR = (BASE_DIR.parent / "data" / "medical_pdfs" / "clinical_books").resolve()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
BATCH_SIZE = 64

MIN_TEXT_LEN = 120
MAX_TEXT_LEN = 3000

EMBED_RETRIES = 2
DB_RETRIES = 2

SLEEP = 0.2

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("menoeaze.clinical_ingest")


# ─────────────────────────────────────────────
# UTIL
# ─────────────────────────────────────────────
def _hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def _is_valid_text(text: str) -> bool:
    if not text or len(text) < MIN_TEXT_LEN:
        return False

    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    return alpha_ratio > 0.65


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").replace("\n", " "))
    return text[:MAX_TEXT_LEN] if _is_valid_text(text) else ""


# ─────────────────────────────────────────────
# PDF LOADER
# ─────────────────────────────────────────────
def load_pdf_pages(path: Path) -> List[str]:
    try:
        reader = PdfReader(str(path))
        pages = []

        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                cleaned = clean_text(txt)
                if cleaned:
                    pages.append(cleaned)

        return pages

    except Exception as e:
        logger.error(f"[PDF ERROR] {path.name}: {e}")
        return []


# ─────────────────────────────────────────────
# SEMANTIC CHUNKING (SENTENCE-AWARE)
# ─────────────────────────────────────────────
def _split_sentences(text: str) -> List[str]:
    return re.split(r'(?<=[.!?])\s+', text)


def chunk_pages(pages: List[str]) -> List[str]:

    chunks = []
    buffer = ""

    for page in pages:
        sentences = _split_sentences(page)

        for sent in sentences:
            if len(buffer) + len(sent) < CHUNK_SIZE:
                buffer += " " + sent
            else:
                if _is_valid_text(buffer):
                    chunks.append(buffer.strip())

                buffer = buffer[-CHUNK_OVERLAP:] + " " + sent

    if _is_valid_text(buffer):
        chunks.append(buffer.strip())

    return chunks


# ─────────────────────────────────────────────
# DEDUP (DB + MEMORY)
# ─────────────────────────────────────────────
def get_existing_hashes(doc_name: str) -> Set[str]:
    rows = fetch_table(
        "medical_documents",
        filters={"document_name": doc_name},
        limit=10000
    )
    return set(r.get("hash") for r in rows)


def is_doc_ingested(doc_name: str) -> bool:
    rows = fetch_table(
        "medical_documents",
        filters={"document_name": doc_name},
        limit=1
    )
    return len(rows) > 0


# ─────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────
def embed_chunks(embedder, chunks: List[str]):

    for attempt in range(EMBED_RETRIES + 1):
        try:
            emb = embedder.embed(chunks)

            if len(emb) != len(chunks):
                raise ValueError("Embedding mismatch")

            if emb.shape[1] != embedder.EXPECTED_DIM:
                raise ValueError("Bad dimension")

            # magnitude check (NEW)
            norms = np.linalg.norm(emb, axis=1)
            if not np.all((norms > 0.5) & (norms < 2.0)):
                raise ValueError("Embedding anomaly detected")

            return emb

        except Exception as e:
            logger.warning(f"[EMBED RETRY {attempt}] {e}")
            time.sleep(0.5)

    return None


# ─────────────────────────────────────────────
# SAFE DB INSERT
# ─────────────────────────────────────────────
def safe_insert(batch: List[Dict]):

    for attempt in range(DB_RETRIES + 1):
        try:
            return insert_bulk("medical_documents", batch)
        except Exception as e:
            logger.warning(f"[DB RETRY {attempt}] {e}")
            time.sleep(0.5)

    return False


# ─────────────────────────────────────────────
# MAIN INGESTION
# ─────────────────────────────────────────────
def run_clinical_ingestion():

    embedder = HFEmbedder()
    pdf_files = list(BOOK_DIR.glob("*.pdf"))

    if not pdf_files:
        logger.error("[ERROR] No clinical books found")
        return

    logger.info(f"[START] {len(pdf_files)} books")

    total_chunks = 0

    for pdf_path in tqdm(pdf_files):

        doc_name = pdf_path.name

        existing_hashes = get_existing_hashes(doc_name)

        pages = load_pdf_pages(pdf_path)

        if not pages:
            logger.warning(f"[EMPTY] {doc_name}")
            continue

        chunks = chunk_pages(pages)

        if not chunks:
            continue

        embeddings = embed_chunks(embedder, chunks)

        if embeddings is None:
            logger.error(f"[FAIL EMBED] {doc_name}")
            continue

        batch = []

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):

            if not np.isfinite(emb).all():
                continue

            h = _hash(chunk)

            if h in existing_hashes:
                continue

            row = {
                "id": str(uuid.uuid4()),
                "title": doc_name,
                "content": chunk,
                "source": "clinical_book",
                "document_name": doc_name,
                "chunk_index": i,
                "embedding": emb.tolist(),
                "hash": h,
                "priority": 1.0,
                "is_deleted": False,
            }

            batch.append(row)
            total_chunks += 1

        if batch:
            success = safe_insert(batch)

            if not success:
                logger.error(f"[FAIL INSERT] {doc_name}")
                continue

        time.sleep(SLEEP)

    logger.info(f"[DONE] total_chunks={total_chunks}")


if __name__ == "__main__":
    run_clinical_ingestion()