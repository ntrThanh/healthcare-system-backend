#!/usr/bin/env python
"""
Utility script: index documents into the RAG vector store.

Usage:
    python scripts/index_documents.py --source ./data/docs --glob "*.txt"
    python scripts/index_documents.py --source ./data/docs --glob "*.json" --field content

Options:
    --source   Directory containing documents
    --glob     File glob pattern (default: *.txt)
    --field    For JSON files: which field holds the text (default: None = whole file)
    --chunk    Chunk size in characters (default: 512)
    --overlap  Overlap between chunks (default: 64)
"""
import argparse
import os
import sys
import uuid
import json
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.core.config import settings
from app.services.rag_service import rag_service


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def main():
    parser = argparse.ArgumentParser(description="Index documents into RAG vector store")
    parser.add_argument("--source", required=True, help="Source directory")
    parser.add_argument("--glob", default="*.txt", help="File glob pattern")
    parser.add_argument("--field", default=None, help="JSON field for text content")
    parser.add_argument("--chunk", type=int, default=512, help="Chunk size in chars")
    parser.add_argument("--overlap", type=int, default=64, help="Chunk overlap in chars")
    args = parser.parse_args()

    source_dir = Path(args.source)
    if not source_dir.exists():
        print(f"[ERROR] Source directory not found: {source_dir}")
        sys.exit(1)

    print(f"Loading RAG service from {settings.EMBEDDING_MODEL_PATH} …")
    rag_service.load()

    files = list(source_dir.glob(args.glob))
    if not files:
        print(f"[WARN] No files matched {args.glob} in {source_dir}")
        sys.exit(0)

    docs = []
    for filepath in files:
        try:
            raw = filepath.read_text(encoding="utf-8")
            if args.field:
                data = json.loads(raw)
                text = data.get(args.field, "")
            else:
                text = raw

            chunks = chunk_text(text, args.chunk, args.overlap)
            for i, chunk in enumerate(chunks):
                docs.append({
                    "id": f"{filepath.stem}__chunk{i}",
                    "text": chunk,
                    "metadata": {
                        "source": str(filepath),
                        "chunk_index": i,
                    },
                })
        except Exception as e:
            print(f"[WARN] Failed to process {filepath}: {e}")

    print(f"Indexing {len(docs)} chunks from {len(files)} files …")
    rag_service.add_documents(docs)
    print(f"Done. Vector store saved to {settings.VECTOR_DB_PATH}")


if __name__ == "__main__":
    main()
