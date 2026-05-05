# ingest_clinical_books.py — PRODUCTION v6 (NON-BLOCKING, NO RPC DEPENDENCY)

import os
import json
import uuid
import logging
import hashlib
import time
import re
from pathlib import Path
from typing import List, Dict, Set, Iterator

import numpy as np
from tqdm import tqdm

# Safely import PyPDF and suppress warnings
logging.getLogger("pypdf").setLevel(logging.ERROR)
from pypdf import PdfReader

from ml_engine.hf_embedder import HFEmbedder
from ml_engine.db_client import insert_bulk, fetch_table

# ─────────────────────────────────────────────
# SENTENCE SPLITTING (NO NLTK)
# ─────────────────────────────────────────────
def sent_tokenize(text: str) -> list:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
BOOK_DIR = (BASE_DIR.parent / "data" / "medical_pdfs" / "clinical_books").resolve()
CHECKPOINT_FILE = (BASE_DIR.parent / "data" / "processed" / "ingest_checkpoint_books.json").resolve()

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
MIN_WORDS_PER_CHUNK = 10

DB_RETRIES = 3

# Adaptive Batching
MIN_BATCH_SIZE = 16
MAX_BATCH_SIZE = 256
INITIAL_BATCH_SIZE = 64

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("menoeaze.ingest.books")

# ─────────────────────────────────────────────
# CHECKPOINTING
# ─────────────────────────────────────────────
def load_checkpoint() -> Dict:
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"processed_docs": [], "last_doc": None, "last_chunk_idx": -1}

def save_checkpoint(doc_name: str, chunk_idx: int, is_complete: bool):
    try:
        CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        ckpt = load_checkpoint()
        if is_complete and doc_name not in ckpt["processed_docs"]:
            ckpt["processed_docs"].append(doc_name)
            ckpt["last_doc"] = None
            ckpt["last_chunk_idx"] = -1
        else:
            ckpt["last_doc"] = doc_name
            ckpt["last_chunk_idx"] = chunk_idx
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(ckpt, f)
    except Exception as e:
        logger.warning(f"[CHECKPOINT] Save failed: {e}")

# ─────────────────────────────────────────────
# UTIL
# ─────────────────────────────────────────────
def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()

def _is_valid_chunk(text: str) -> bool:
    if not text:
        return False
    words = text.split()
    if len(words) < MIN_WORDS_PER_CHUNK:
        return False
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    return alpha_ratio > 0.5

# ─────────────────────────────────────────────
# STREAMING PDF EXTRACTION & CHUNKING
# ─────────────────────────────────────────────
def yield_chunks(path: Path) -> Iterator[str]:
    try:
        reader = PdfReader(str(path), strict=False)
        current_chunk = []
        current_len = 0

        for page in reader.pages:
            try:
                page_text = page.extract_text()
                if not page_text:
                    continue
                page_text = clean_text(page_text)
                sentences = sent_tokenize(page_text)

                for sentence in sentences:
                    s_len = len(sentence)
                    if current_len + s_len > CHUNK_SIZE:
                        chunk_str = " ".join(current_chunk)
                        if _is_valid_chunk(chunk_str):
                            yield chunk_str
                        overlap = []
                        overlap_len = 0
                        for s in reversed(current_chunk):
                            overlap.insert(0, s)
                            overlap_len += len(s)
                            if overlap_len >= CHUNK_OVERLAP:
                                break
                        current_chunk = overlap
                        current_len = sum(len(s) for s in overlap)
                    current_chunk.append(sentence)
                    current_len += s_len
            except Exception as e:
                logger.debug(f"Skipped corrupted page in {path.name}: {e}")
                continue

        if current_chunk:
            chunk_str = " ".join(current_chunk)
            if _is_valid_chunk(chunk_str):
                yield chunk_str
    except Exception as e:
        logger.error(f"[PDF FAIL] {path.name} | {e}")

# ─────────────────────────────────────────────
# VERIFICATION (HASH-ONLY — NO RPC)
# ─────────────────────────────────────────────
def verify_hashes_exist(hashes: List[str]) -> float:
    if not hashes:
        return 1.0
    test_hashes = list(set([hashes[0], hashes[-1]]))
    found = 0
    for h in test_hashes:
        try:
            rows = fetch_table("medical_documents", filters={"hash": h}, limit=1)
            if rows:
                found += 1
        except Exception:
            pass
    return found / len(test_hashes)

# ─────────────────────────────────────────────
# EMBEDDING (SAFE — DIRECT CALL)
# ─────────────────────────────────────────────
def safe_embed(embedder, chunks: List[str]) -> np.ndarray:
    try:
        emb = embedder.embed(chunks)
        if emb is not None and len(emb) == len(chunks) and np.isfinite(emb).all():
            return emb
    except Exception as e:
        logger.warning(f"[EMBED FAIL] {e}")
    return None

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_clinical_ingestion():
    embedder = HFEmbedder()
    embedder._load_model()

    pdf_files = list(BOOK_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.error("[ERROR] No clinical books found in %s", BOOK_DIR)
        return

    logger.info(f"[START] {len(pdf_files)} books in {BOOK_DIR}")

    ckpt = load_checkpoint()

    metrics = {
        "docs_processed": 0,
        "docs_skipped": 0,
        "docs_failed": 0,
        "total_chunks_extracted": 0,
        "total_chunks_inserted": 0,
        "embed_failures": 0,
        "db_failures": 0,
        "start_time": time.time()
    }

    current_batch_size = INITIAL_BATCH_SIZE

    for pdf_path in tqdm(pdf_files, desc="Books"):
        doc_name = pdf_path.name

        if doc_name in ckpt["processed_docs"]:
            metrics["docs_skipped"] += 1
            continue

        resume_idx = 0
        if ckpt["last_doc"] == doc_name:
            resume_idx = ckpt["last_chunk_idx"] + 1
            logger.info(f"Resuming {doc_name} from chunk {resume_idx}")

        chunk_generator = yield_chunks(pdf_path)

        batch_chunks = []
        batch_indices = []
        global_chunk_idx = 0
        doc_inserted = 0
        doc_failed = False

        for chunk_text in chunk_generator:
            if global_chunk_idx < resume_idx:
                global_chunk_idx += 1
                continue

            batch_chunks.append(chunk_text)
            batch_indices.append(global_chunk_idx)
            global_chunk_idx += 1
            metrics["total_chunks_extracted"] += 1

            if len(batch_chunks) >= current_batch_size:
                # ── EMBED ──
                embeddings = safe_embed(embedder, batch_chunks)
                if embeddings is None:
                    logger.error(f"[EMBED] Failed for batch in {doc_name}. Skipping batch.")
                    metrics["embed_failures"] += len(batch_chunks)
                    batch_chunks = []
                    batch_indices = []
                    continue

                # ── BUILD PAYLOADS ──
                payloads = []
                batch_hashes = []
                for txt, emb, idx in zip(batch_chunks, embeddings, batch_indices):
                    h = _hash(txt)
                    batch_hashes.append(h)
                    payloads.append({
                        "id": str(uuid.uuid4()),
                        "title": doc_name,
                        "content": txt,
                        "source": "clinical_book",
                        "document_name": doc_name,
                        "chunk_index": idx,
                        "embedding": emb.tolist(),
                        "hash": h,
                        "priority": 1.0,
                        "is_deleted": False,
                    })

                # ── DB INSERT ──
                insert_start = time.time()
                success = False
                for att in range(DB_RETRIES):
                    try:
                        if insert_bulk("medical_documents", payloads):
                            success = True
                            break
                    except Exception as e:
                        logger.warning(f"[DB] Insert attempt {att+1} failed: {e}")
                        time.sleep(0.5)

                insert_latency = time.time() - insert_start

                if success:
                    try:
                        verif = verify_hashes_exist(batch_hashes)
                        if verif < 1.0:
                            logger.warning(f"Read-after-write: {verif*100:.0f}%")
                    except Exception:
                        pass

                    doc_inserted += len(payloads)
                    metrics["total_chunks_inserted"] += len(payloads)

                    if insert_latency > 2.0:
                        current_batch_size = max(MIN_BATCH_SIZE, current_batch_size // 2)
                    elif insert_latency < 0.5:
                        current_batch_size = min(MAX_BATCH_SIZE, current_batch_size + 16)

                    save_checkpoint(doc_name, batch_indices[-1], is_complete=False)
                    logger.info(f"  [{doc_name}] +{len(payloads)} chunks ({doc_inserted} total) | {insert_latency:.1f}s | batch={current_batch_size}")
                else:
                    logger.error(f"[DB] Insert failed for batch in {doc_name}. Continuing...")
                    metrics["db_failures"] += len(payloads)

                batch_chunks = []
                batch_indices = []
                time.sleep(0.05)

        # ── REMAINDER ──
        if batch_chunks and not doc_failed:
            embeddings = safe_embed(embedder, batch_chunks)
            if embeddings is not None:
                payloads = []
                for txt, emb, idx in zip(batch_chunks, embeddings, batch_indices):
                    payloads.append({
                        "id": str(uuid.uuid4()),
                        "title": doc_name,
                        "content": txt,
                        "source": "clinical_book",
                        "document_name": doc_name,
                        "chunk_index": idx,
                        "embedding": emb.tolist(),
                        "hash": _hash(txt),
                        "priority": 1.0,
                        "is_deleted": False,
                    })
                try:
                    if insert_bulk("medical_documents", payloads):
                        doc_inserted += len(payloads)
                        metrics["total_chunks_inserted"] += len(payloads)
                    else:
                        metrics["db_failures"] += len(payloads)
                except Exception as e:
                    logger.warning(f"[DB] Remainder insert failed: {e}")
                    metrics["db_failures"] += len(payloads)
            else:
                metrics["embed_failures"] += len(batch_chunks)

        if not doc_failed:
            metrics["docs_processed"] += 1
            save_checkpoint(doc_name, -1, is_complete=True)
            logger.info(f"✅ Completed {doc_name}: {doc_inserted} chunks inserted.")
        else:
            metrics["docs_failed"] += 1

    # ── FINAL REPORT ──
    duration = time.time() - metrics["start_time"]
    logger.info("=" * 50)
    logger.info("=== CLINICAL BOOKS INGESTION REPORT ===")
    logger.info(f"Total Duration : {duration:.1f}s")
    logger.info(f"Books Processed: {metrics['docs_processed']}")
    logger.info(f"Books Skipped  : {metrics['docs_skipped']}")
    logger.info(f"Books Failed   : {metrics['docs_failed']}")
    logger.info(f"Chunks Extracted: {metrics['total_chunks_extracted']}")
    logger.info(f"Chunks Inserted: {metrics['total_chunks_inserted']}")
    logger.info(f"Embed Failures : {metrics['embed_failures']}")
    logger.info(f"DB Failures    : {metrics['db_failures']}")
    if metrics["total_chunks_extracted"] > 0:
        success_rate = (metrics["total_chunks_inserted"] / metrics["total_chunks_extracted"]) * 100
        logger.info(f"Success Rate   : {success_rate:.2f}%")
    logger.info("=" * 50)

if __name__ == "__main__":
    run_clinical_ingestion()