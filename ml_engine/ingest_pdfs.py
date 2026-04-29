"""
ingest_pdfs.py
==============
Standalone script to ingest medical PDFs into ChromaDB.

Usage:
    python ingest_pdfs.py              # ingest (skip if already done)
    python ingest_pdfs.py --force      # force re-ingestion

Reads all PDFs from ../public/pdfs/ and chunks them into
the ChromaDB persistent collection with source metadata.

Safe to re-run: uses upsert (no duplicates).
"""

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    print("━" * 60)
    print("  PDF Ingestion → ChromaDB")
    print("━" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.abspath(os.path.join(base_dir, "..", "public", "pdfs"))

    if not os.path.isdir(pdf_dir):
        print(f"ERROR: PDF directory not found: {pdf_dir}")
        sys.exit(1)

    pdf_count = len([f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")])
    print(f"PDF directory : {pdf_dir}")
    print(f"PDFs found    : {pdf_count}")

    force = "--force" in sys.argv
    print(f"Mode          : {'FORCE re-ingestion' if force else 'Skip if exists'}")
    print()

    from ml_engine.rag_engine import ingest_pdfs
    result = ingest_pdfs(pdf_dir, force=force)

    print()
    print("━" * 60)
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        sys.exit(1)
    elif result.get("skipped"):
        print(f"  Skipped — collection already has {result['total_chunks']} chunks")
        print("  Use --force to re-ingest")
    else:
        print(f"  PDFs processed : {result['total_pdfs']}")
        print(f"  Total chunks   : {result['total_chunks']}")
    print("━" * 60)


if __name__ == "__main__":
    main()
