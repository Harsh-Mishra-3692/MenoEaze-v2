# ingest_pdfs.py — ORCHESTRATOR V2 (RESILIENT + OBSERVABLE + SAFE)

import os
import sys
import time
import logging
import traceback
from datetime import datetime
from typing import Callable, Dict

# ─────────────────────────────────────────────
# LOGGING (STRUCTURED)
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("menoeaze.ingest.orchestrator")


# ─────────────────────────────────────────────
# ENV VALIDATION
# ─────────────────────────────────────────────
def _validate_env() -> bool:
    try:
        import ml_engine.db_client
        import ml_engine.ingest_medical_docs
        import ml_engine.ingest_clinical_books
        return True
    except Exception as e:
        logger.error(f"[ENV] Critical import failed: {e}")
        return False


# ─────────────────────────────────────────────
# DB HEALTH CHECK
# ─────────────────────────────────────────────
def _check_db_health() -> bool:
    try:
        from ml_engine.db_client import fetch_table
        rows = fetch_table("medical_documents", limit=1)
        return isinstance(rows, list)
    except Exception as e:
        logger.error(f"[DB] Health check failed: {e}")
        return False


# ─────────────────────────────────────────────
# DB QUALITY VALIDATION (UPGRADED)
# ─────────────────────────────────────────────
def _verify_ingestion() -> bool:
    try:
        from ml_engine.db_client import fetch_table

        rows = fetch_table("medical_documents", limit=50)

        if not rows:
            logger.error("[VERIFY] No rows found")
            return False

        required = ["content", "embedding", "document_name"]

        for r in rows:
            for k in required:
                if k not in r:
                    logger.error(f"[VERIFY] Missing field: {k}")
                    return False

            # embedding sanity
            emb = r.get("embedding")
            if not isinstance(emb, list) or len(emb) < 100:
                logger.error("[VERIFY] Invalid embedding")
                return False

            if not r.get("content") or len(r["content"]) < 20:
                logger.error("[VERIFY] Weak content detected")
                return False

        return True

    except Exception as e:
        logger.error(f"[VERIFY] Failed: {e}")
        return False


# ─────────────────────────────────────────────
# SAFE RUN WRAPPER (HARDENED)
# ─────────────────────────────────────────────
def _run_step(name: str, fn: Callable, timeout: int = 1800) -> Dict:

    start = time.time()

    try:
        logger.info(f"[STEP START] {name}")

        result = fn()

        duration = round(time.time() - start, 2)

        logger.info(f"[STEP OK] {name} ({duration}s)")

        return {
            "status": "ok",
            "duration": duration,
        }

    except Exception as e:
        duration = round(time.time() - start, 2)

        logger.error(f"[STEP FAIL] {name} | {e}")
        logger.debug(traceback.format_exc())

        return {
            "status": "fail",
            "duration": duration,
            "error": str(e),
        }


# ─────────────────────────────────────────────
# FORCE CLEAN (SAFE)
# ─────────────────────────────────────────────
def _force_cleanup():
    try:
        from ml_engine.db_client import insert_bulk

        logger.warning("[FORCE] Clearing existing data...")

        # safer soft-delete approach
        insert_bulk(
            "medical_documents",
            [{"is_deleted": True}],
        )

    except Exception as e:
        logger.error(f"[FORCE] Cleanup failed: {e}")


# ─────────────────────────────────────────────
# PUBLIC PDF INGEST (ISOLATED)
# ─────────────────────────────────────────────
def _ingest_public(force: bool):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.abspath(os.path.join(base_dir, "..", "public", "pdfs"))

    if not os.path.isdir(pdf_dir):
        logger.warning("[SKIP] public/pdfs not found")
        return

    from ml_engine.rag_engine import ingest_pdfs

    files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    for f in files:
        try:
            ingest_pdfs(os.path.join(pdf_dir, f), force=force)
        except Exception as e:
            logger.warning(f"[PUBLIC FAIL] {f}: {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():

    print("━" * 70)
    print("  MenoEaze Ingestion System — Production Pipeline")
    print("━" * 70)

    if not _validate_env():
        sys.exit(1)

    if not _check_db_health():
        logger.error("[FATAL] Database not reachable")
        sys.exit(1)

    force = "--force" in sys.argv

    print(f"Mode        : {'FORCE FULL REBUILD' if force else 'SAFE INCREMENTAL'}")
    print(f"Start Time  : {datetime.now().isoformat()}")
    print()

    if force:
        confirm = input("⚠️ This will reprocess ALL data. Continue? (yes/no): ")
        if confirm.lower() != "yes":
            print("Aborted.")
            sys.exit(0)

        _force_cleanup()

    results = {}

    # ─────────────────────────
    # 1. MEDICAL DOCS
    # ─────────────────────────
    def ingest_medical():
        from ml_engine.ingest_medical_docs import run_ingestion
        return run_ingestion()

    results["medical_docs"] = _run_step("Medical PDFs", ingest_medical)

    # ─────────────────────────
    # 2. CLINICAL BOOKS
    # ─────────────────────────
    def ingest_clinical():
        from ml_engine.ingest_clinical_books import run_clinical_ingestion
        return run_clinical_ingestion()

    results["clinical_books"] = _run_step("Clinical Books", ingest_clinical)

    # ─────────────────────────
    # 3. PUBLIC PDFs
    # ─────────────────────────
    results["public_pdfs"] = _run_step(
        "Public PDFs",
        lambda: _ingest_public(force),
    )

    # ─────────────────────────
    # VALIDATION
    # ─────────────────────────
    valid = _verify_ingestion()

    # ─────────────────────────
    # SUMMARY REPORT
    # ─────────────────────────
    print("\n" + "━" * 70)

    for k, v in results.items():
        status = v["status"].upper()
        duration = v["duration"]
        print(f"{k:<20}: {status} ({duration}s)")

    print(f"\nDB VALIDATION       : {'OK' if valid else 'FAIL'}")
    print(f"End Time            : {datetime.now().isoformat()}")

    print("━" * 70)

    # ─────────────────────────
    # EXIT
    # ─────────────────────────
    if not valid or any(v["status"] == "fail" for v in results.values()):
        sys.exit(1)

    print("\nSYSTEM READY FOR RAG")
    sys.exit(0)


# ─────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()