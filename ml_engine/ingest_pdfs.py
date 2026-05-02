# ingest_pdfs.py — SYSTEM ORCHESTRATOR (FINAL)

import os
import sys
import logging
import traceback
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("menoeaze.ingest.orchestrator")


# ─────────────────────────────────────────────
# ENV VALIDATION
# ─────────────────────────────────────────────
def _validate_env():
    try:
        import ml_engine.rag_engine
        import ml_engine.hybrid_retriever
        import ml_engine.db_client
        return True
    except Exception as e:
        logger.error(f"[ENV] Critical import failed: {e}")
        return False


# ─────────────────────────────────────────────
# DB HEALTH CHECK
# ─────────────────────────────────────────────
def _check_db_health():
    try:
        from ml_engine.db_client import fetch_table

        rows = fetch_table("medical_documents", limit=1)
        return isinstance(rows, list)

    except Exception as e:
        logger.error(f"[DB] Health check failed: {e}")
        return False


# ─────────────────────────────────────────────
# VERIFY INGESTION OUTPUT
# ─────────────────────────────────────────────
def _verify_ingestion():
    try:
        from ml_engine.db_client import fetch_table

        rows = fetch_table("medical_documents", limit=10)

        if not rows:
            return False

        # minimal schema validation
        required = ["content", "embedding", "document_name"]

        for r in rows:
            for k in required:
                if k not in r:
                    return False

        return True

    except Exception:
        return False


# ─────────────────────────────────────────────
# SAFE RUN WRAPPER
# ─────────────────────────────────────────────
def _run_step(name, fn):

    try:
        logger.info(f"[STEP START] {name}")
        result = fn()

        logger.info(f"[STEP OK] {name}")
        return True

    except Exception as e:
        logger.error(f"[STEP FAIL] {name} | {e}")
        logger.debug(traceback.format_exc())
        return False


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():

    print("━" * 70)
    print("  MenoEaze Ingestion System (Unified Pipeline)")
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
        confirm = input("⚠️  This will re-ingest ALL sources. Continue? (yes/no): ")
        if confirm.lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    success_map = {}

    # ─────────────────────────
    # 1. MEDICAL DOCS (PDFs)
    # ─────────────────────────
    def ingest_medical():
        from ml_engine.ingest_medical_docs import run_ingestion
        run_ingestion()

    success_map["medical_docs"] = _run_step("Medical PDFs", ingest_medical)

    # ─────────────────────────
    # 2. CLINICAL BOOKS
    # ─────────────────────────
    def ingest_clinical():
        from ml_engine.ingest_clinical_books import run_clinical_ingestion
        run_clinical_ingestion()

    success_map["clinical_books"] = _run_step("Clinical Books", ingest_clinical)

    # ─────────────────────────
    # 3. LEGACY PUBLIC PDFs (OPTIONAL)
    # ─────────────────────────
    def ingest_public():
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_dir = os.path.abspath(os.path.join(base_dir, "..", "public", "pdfs"))

        if not os.path.isdir(pdf_dir):
            logger.warning("[SKIP] public/pdfs not found")
            return

        from ml_engine.rag_engine import ingest_pdfs

        files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

        for f in files:
            ingest_pdfs(os.path.join(pdf_dir, f), force=force)

    success_map["public_pdfs"] = _run_step("Public PDFs", ingest_public)

    # ─────────────────────────
    # 4. POST-INGEST VALIDATION
    # ─────────────────────────
    valid = _verify_ingestion()

    # ─────────────────────────
    # SUMMARY
    # ─────────────────────────
    print()
    print("━" * 70)

    for k, v in success_map.items():
        print(f"{k:<20}: {'OK' if v else 'FAIL'}")

    print(f"\nDB VALIDATION       : {'OK' if valid else 'FAIL'}")
    print(f"End Time            : {datetime.now().isoformat()}")

    print("━" * 70)

    # ─────────────────────────
    # EXIT
    # ─────────────────────────
    if not all(success_map.values()) or not valid:
        sys.exit(1)

    print("\n✅ SYSTEM READY FOR RAG")
    sys.exit(0)


# ─────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()