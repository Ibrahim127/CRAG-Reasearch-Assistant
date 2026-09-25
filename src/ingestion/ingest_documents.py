"""
Generalized document ingestion pipeline using LiteParse (local) with Parallel Processing.

Accepts ANY folder of PDFs and a collection name -- point it at a 
folder and collection name, and you get an independently queryable 
knowledge base processed concurrently on your local machine.

Usage:
    python ingest_documents.py --pdf_dir path/to/pdfs --collection_name my_docs

Pipeline: PDF -> LiteParse (local, parallel) -> markdown -> section-aware chunk ->
embed (BGE) -> write to a NEW or EXISTING Chroma collection.
"""

import argparse
import asyncio
import os
import re
import sys
import warnings
from pathlib import Path
from tqdm import tqdm

# Suppress common non-critical library warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------
# Preflight Checks
# ---------------------------------------------------------------------

def check_gpu_available() -> None:
    """
    Informational check only -- LiteParse extraction runs purely on local CPU,
    while the BGE embedding step benefits from GPU if present.
    """
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[+] GPU detected: {torch.cuda.get_device_name(0)} (used for embedding)")
        else:
            print("[*] No GPU detected -- embedding runs on CPU. LiteParse extraction is CPU-native.")
    except ImportError:
        print("[*] PyTorch not installed -- needed for the embedding step.")


# ---------------------------------------------------------------------
# Stage 1: PDF -> LiteParse (local, async/parallel) -> Markdown
# ---------------------------------------------------------------------

_LITEPARSE_PARSER = None


def get_liteparse_parser(num_workers: int = 4):
    """Lazily initializes a local LiteParse worker pool."""
    global _LITEPARSE_PARSER
    if _LITEPARSE_PARSER is not None:
        return _LITEPARSE_PARSER

    try:
        from liteparse import LiteParse
    except ImportError:
        print("[!] liteparse not installed (`pip install liteparse`). Falling back to pypdf/pdfplumber.")
        return None

    print(f"[*] Initializing local LiteParse with {num_workers} concurrent workers...")
    
    # Initialize the local parser pool to process documents in parallel
    # output_format="markdown" forces it to reconstruct tables and headers
    _LITEPARSE_PARSER = LiteParse(
        pool_size=num_workers,
        output_format="markdown",
        ocr_enabled=True,
        parse_timeout=45
    )
    _LITEPARSE_PARSER.warm_up()
    
    return _LITEPARSE_PARSER


def fallback_pdf_extraction(pdf_path: Path) -> str:
    """Local fallback text extractor, used if LiteParse is unavailable or fails."""
    text_parts = []

    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages):
            txt = page.extract_text()
            if txt:
                text_parts.append(f"## Page {i+1}\n\n{txt}")
        if text_parts:
            return "\n\n".join(text_parts)
    except Exception:
        pass

    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                txt = page.extract_text()
                if txt:
                    text_parts.append(f"## Page {i+1}\n\n{txt}")
        if text_parts:
            return "\n\n".join(text_parts)
    except Exception:
        pass

    return ""


async def process_single_pdf_async(pdf_path: Path, parser, semaphore, index: int, total: int, progress_callback=None) -> tuple[list[dict], list[str], dict]:
    """Processes a single PDF asynchronously using a semaphore to cap concurrency."""
    async with semaphore:
        doc_id = pdf_path.stem
        print(f"[*] Processing {pdf_path.name}...")
        
        if progress_callback:
            progress_callback(index / total, f"Processing {pdf_path.name}...")

        md_text = ""
        try:
            if parser is not None:
                # Run the local worker pool call in a separate thread so it doesn't block the event loop
                result = await asyncio.to_thread(parser.parse, str(pdf_path))
                md_text = result.text
            
            if not md_text or not md_text.strip():
                md_text = fallback_pdf_extraction(pdf_path)
        except Exception as e:
            print(f"    [!] LiteParse failed on {pdf_path.name}: {e}. Falling back to standard pypdf extraction...")
            md_text = fallback_pdf_extraction(pdf_path)

        if not md_text.strip():
            raise RuntimeError(f"Failed to extract text from {pdf_path.name} via LiteParse and local fallbacks.")

        sections, tables = parse_markdown_sections(md_text)
        title = sections[0]["text"].split("\n")[0][:120] if sections else doc_id
        chunks = build_chunks_for_document(doc_id, title, sections)

        manifest_item = {
            "doc_id": doc_id,
            "filename": pdf_path.name,
            "title": title,
            "num_sections": len(sections),
            "num_chunks": len(chunks),
            "num_tables_excluded": len(tables),
        }
        print(f"    -> {pdf_path.name}: {len(sections)} sections, {len(chunks)} chunks, {len(tables)} table(s) excluded")
        
        return chunks, tables, manifest_item


# ---------------------------------------------------------------------
# Stage 2: Markdown -> Sections -> Chunks
# ---------------------------------------------------------------------

TABLE_PATTERN = re.compile(r"\\begin\{table\}.*?\\end\{table\}", re.DOTALL)


def parse_markdown_sections(md_text: str) -> tuple[list[dict], list[str]]:
    tables = TABLE_PATTERN.findall(md_text)
    text_without_tables = TABLE_PATTERN.sub("", md_text)

    lines = text_without_tables.split("\n")
    sections = []
    current_name = "Title"
    current_lines = []
    heading_re = re.compile(r"^#{1,6}\s+(.+)$")
    hit_references = False

    for line in lines:
        m = heading_re.match(line.strip())
        if m:
            heading_text = m.group(1).strip()
            if heading_text.lower() in ("references", "bibliography"):
                hit_references = True
            if hit_references:
                continue
            if current_lines:
                sections.append({
                    "section_name": current_name,
                    "text": "\n".join(current_lines).strip(),
                })
            current_name = heading_text
            current_lines = []
        elif not hit_references:
            current_lines.append(line)

    if current_lines and not hit_references:
        sections.append({
            "section_name": current_name,
            "text": "\n".join(current_lines).strip(),
        })

    sections = [s for s in sections if len(s["text"].strip()) > 10]

    if not sections and md_text.strip():
        sections = [{
            "section_name": "Full Document",
            "text": md_text.strip()
        }]

    return sections, tables


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def build_chunks_for_document(doc_id: str, title: str, sections: list[dict]) -> list[dict]:
    chunks = []
    global_chunk_index = 0  # Continuous counter for the entire document guarantees unique IDs
    
    for section in sections:
        pieces = chunk_text(section["text"])
        for piece in pieces:
            # Sanitize section name for readability in the ID
            safe_section = re.sub(r"[^a-zA-Z0-9]+", "_", section["section_name"])[:30]
            
            chunks.append({
                "id": f"{doc_id}_{safe_section}_{global_chunk_index}",
                "text": piece,
                "metadata": {
                    "source_paper": doc_id,
                    "title": title,
                    "section_name": section["section_name"],
                    "content_type": "text",
                    "chunk_index": global_chunk_index,
                },
            })
            global_chunk_index += 1  # Increment for every chunk, never resetting
            
    return chunks


# ---------------------------------------------------------------------
# Stage 3: Embed + Write to Chroma
# ---------------------------------------------------------------------

def embed_and_store(chunks: list[dict], collection_name: str, chroma_path: str, append: bool = False) -> None:
    import chromadb
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading embedding model (BAAI/bge-small-en-v1.5) on device '{device}'...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

    print(f"[*] Embedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    ).tolist()

    client = chromadb.PersistentClient(path=chroma_path)

    if not append:
        try:
            client.delete_collection(collection_name)
            print(f"[*] Deleted existing collection '{collection_name}'")
        except Exception:
            pass
        collection = client.create_collection(collection_name)
    else:
        collection = client.get_or_create_collection(collection_name)
        print(f"[*] Appending to existing collection '{collection_name}'")

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    batch_size = 500
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"[+] Collection '{collection_name}' now has {collection.count()} chunks total")


# ---------------------------------------------------------------------
# Orchestration (Async / Parallel Execution)
# ---------------------------------------------------------------------

async def ingest_pdf_folder_async(pdf_dir: str, collection_name: str, chroma_path: str, append: bool = False, progress_callback=None, max_concurrent: int = 4) -> dict:
    pdf_dir_path = Path(pdf_dir)
    pdf_files = sorted(pdf_dir_path.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")

    print(f"[*] Found {len(pdf_files)} PDF(s) in {pdf_dir}. Processing locally in parallel...")

    # Load local LiteParse with the requested concurrency pool size
    parser = get_liteparse_parser(num_workers=max_concurrent)
    
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        process_single_pdf_async(pdf_path, parser, semaphore, i, len(pdf_files), progress_callback)
        for i, pdf_path in enumerate(pdf_files)
    ]
    
    results = await asyncio.gather(*tasks)

    all_chunks = []
    all_tables = []
    manifest = []

    for chunks, tables, manifest_item in results:
        all_chunks.extend(chunks)
        all_tables.extend(tables)
        manifest.append(manifest_item)

    if parser is not None and hasattr(parser, "close"):
        parser.close()

    if progress_callback:
        progress_callback(1.0, "Generating embeddings and saving to ChromaDB...")

    if not all_chunks:
        raise ValueError("No text could be extracted from any PDF")

    embed_and_store(all_chunks, collection_name, chroma_path, append=append)

    return {
        "collection_name": collection_name,
        "total_documents": len(pdf_files),
        "total_chunks": len(all_chunks),
        "total_tables_excluded": len(all_tables),
        "manifest": manifest,
    }


def ingest_pdf_folder(pdf_dir: str, collection_name: str, chroma_path: str, append: bool = False, progress_callback=None) -> dict:
    check_gpu_available()
    return asyncio.run(ingest_pdf_folder_async(pdf_dir, collection_name, chroma_path, append=append, progress_callback=progress_callback, max_concurrent=4))


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a folder of PDFs into a queryable Chroma collection via local LiteParse."
    )
    parser.add_argument("--pdf_dir", required=True, help="Folder containing PDF files")
    parser.add_argument("--collection_name", required=True, help="Name for the Chroma collection")
    parser.add_argument("--chroma_path", default="./chroma_db", help="Path to persistent Chroma store")
    parser.add_argument("--append", action="store_true", help="Append to an existing collection instead of replacing it")
    args = parser.parse_args()

    result = ingest_pdf_folder(args.pdf_dir, args.collection_name, args.chroma_path, append=args.append)

    print("\n" + "=" * 60)
    print(f"DONE: '{result['collection_name']}' -- {result['total_documents']} docs, "
          f"{result['total_chunks']} chunks, {result['total_tables_excluded']} table(s) excluded")
    print("=" * 60)


if __name__ == "__main__":
    main()