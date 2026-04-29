# ingest_medical_docs.py — ELITE (PRODUCTION + SAFE + ARCH-CLEAN)

import uuid
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
from tqdm import tqdm
from pypdf import PdfReader

from ml_engine.hf_embedder import HFEmbedder
from ml_engine.db_client import insert_bulk, fetch_table

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = (BASE_DIR.parent / "data" / "medical_pdfs").resolve()
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
BATCH_SIZE = 200
MIN_DOC_LENGTH = 50

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("menoeaze.ingestion")


# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.replace("\n", " ").split())


# ─────────────────────────────────────────────
# LOAD PDF
# ─────────────────────────────────────────────
def load_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        pages = []

        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                pages.append(txt)

        return clean_text(" ".join(pages))

    except Exception as e:
        logger.error(f"[INGEST] Failed reading {path.name}: {e}")
        return ""


# ─────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────
def chunk_text(text: str) -> List[str]:
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + CHUNK_SIZE
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ─────────────────────────────────────────────
# DEDUP CHECK
# ─────────────────────────────────────────────
def is_already_ingested(doc_name: str) -> bool:
    try:
        rows = fetch_table(
            table="medical_documents",
            limit=1,
            filters={"document_name": doc_name}
        )
        return len(rows) > 0
    except Exception as e:
        logger.warning(f"[INGEST] Dedup check failed: {e}")
        return False


# ─────────────────────────────────────────────
# MAIN INGESTION PIPELINE
# ─────────────────────────────────────────────
def run_ingestion():
    embedder = HFEmbedder()

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        logger.error(f"[INGEST] No PDFs found in {PDF_DIR}")
        return

    logger.info(f"[INGEST] Found {len(pdf_files)} PDFs")

    total_chunks = 0
    batch: List[Dict] = []

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):

        doc_name = pdf_path.name

        # ── Deduplication ────────────────────
        if is_already_ingested(doc_name):
            logger.info(f"[INGEST] Skipping (already ingested): {doc_name}")
            continue

        logger.info(f"[INGEST] Processing: {doc_name}")

        text = load_pdf(pdf_path)

        if not text or len(text) < MIN_DOC_LENGTH:
            logger.warning(f"[INGEST] Skipping short/empty doc: {doc_name}")
            continue

        chunks = chunk_text(text)

        if not chunks:
            logger.warning(f"[INGEST] No valid chunks: {doc_name}")
            continue

        # ── Embedding ────────────────────────
        try:
            embeddings = embedder.embed(chunks)
        except Exception as e:
            logger.error(f"[INGEST] Embedding failed: {e}")
            continue

        if len(embeddings) != len(chunks):
            logger.error("[INGEST] Embedding mismatch")
            continue

        # ── Build rows ───────────────────────
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):

            if emb is None or not isinstance(emb, np.ndarray):
                continue

            row = {
                "id": str(uuid.uuid4()),
                "title": doc_name,
                "content": chunk,
                "source": "pdf",
                "document_name": doc_name,
                "chunk_index": i,
                "embedding": emb.tolist(),
                "is_deleted": False
            }

            batch.append(row)
            total_chunks += 1

            # ── Batch flush ───────────────────
            if len(batch) >= BATCH_SIZE:
                success = insert_bulk("medical_documents", batch)

                if success:
                    logger.info(f"[INGEST] Inserted batch ({len(batch)})")
                else:
                    logger.error("[INGEST] Batch insert failed")

                batch = []

    # ── Final flush ─────────────────────────
    if batch:
        success = insert_bulk("medical_documents", batch)
        if success:
            logger.info(f"[INGEST] Final batch inserted ({len(batch)})")

    logger.info(f"[INGEST] COMPLETE | total_chunks={total_chunks}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_ingestion()