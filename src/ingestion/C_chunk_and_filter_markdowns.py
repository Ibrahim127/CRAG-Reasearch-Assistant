import re
import json
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

MD_DIR = Path("./data/markdown_artifacts")
OUTPUT_JSON = Path("./data/processed_chunks.json")
TABLES_LOG_JSON = Path("./data/excluded_tables_log.json")

REFERENCES_REGEX = re.compile(
    r"^(#+|\*\*)\s*(References|Bibliography|Works Cited)\b.*$", 
    re.IGNORECASE | re.MULTILINE
)

def extract_and_strip_tables(raw_text: str):
    """
    Robust table extractor that handles both properly closed \\end{table} 
    and unclosed/malformed table blocks from OCR output.
    """
    # Regex to find any \\begin{table...} block up to \\end{table...} 
    # OR up to a major Markdown header / double newline if \\end is missing.
    table_pattern = re.compile(
        r"(\\begin\{table\*?\}.*?(?:\\end\{table\*?\}|(?=\n#{1,6}\s)|\Z))",
        re.DOTALL
    )

    tables = []
    # Find all table matches
    for match in table_pattern.finditer(raw_text):
        table_str = match.group(0).strip()
        if table_str:
            tables.append(table_str)

    # Replace all table occurrences with empty string in prose
    prose_only = table_pattern.sub("", raw_text)
    
    # Clean up double empty lines left behind by removed tables
    prose_only = re.sub(r"\n{3,}", "\n\n", prose_only).strip()

    return prose_only, tables

def process_file(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 1. Strip References
    ref_match = REFERENCES_REGEX.search(raw_text)
    if ref_match:
        raw_text = raw_text[:ref_match.start()].strip()

    # 2. HARD EXTRACTION: Pull tables completely out of raw_text BEFORE any chunking
    prose_text, raw_tables = extract_and_strip_tables(raw_text)

    # Package isolated tables
    table_chunks = []
    for tbl in raw_tables:
        table_chunks.append({
            "text": tbl,
            "metadata": {
                "source_paper": file_path.stem,
                "section_name": "Table Block",
                "content_type": "table"
            }
        })

    # 3. Chunk ONLY the remaining clean prose text
    headers_to_split = [("#" * i, f"Header_{i}") for i in range(1, 7)]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split, 
        strip_headers=False
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )

    header_docs = md_splitter.split_text(prose_text)
    text_chunks = []

    for doc in header_docs:
        section_name = "General"
        for i in range(6, 0, -1):
            if f"Header_{i}" in doc.metadata:
                section_name = doc.metadata[f"Header_{i}"]
                break

        sub_chunks = text_splitter.split_text(doc.page_content)
        for sub in sub_chunks:
            text_chunks.append({
                "text": sub.strip(),
                "metadata": {
                    "source_paper": file_path.stem,
                    "section_name": section_name,
                    "content_type": "text"
                }
            })

    return text_chunks, table_chunks

# --- Execution ---
all_text_chunks = []
all_table_chunks = []

for md_file in MD_DIR.glob("*.md"):
    text_chunks, table_chunks = process_file(md_file)
    
    for idx, c in enumerate(text_chunks):
        c["id"] = f"{md_file.stem}_chunk_{idx}"
        c["metadata"]["chunk_index"] = idx
        all_text_chunks.append(c)

    for idx, t in enumerate(table_chunks):
        t["id"] = f"{md_file.stem}_table_{idx}"
        t["metadata"]["chunk_index"] = idx
        all_table_chunks.append(t)

# Output clean prose chunks for ChromaDB
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(all_text_chunks, f, indent=2, ensure_ascii=False)

# Output isolated table log
with open(TABLES_LOG_JSON, "w", encoding="utf-8") as f:
    json.dump(all_table_chunks, f, indent=2, ensure_ascii=False)

print(f"[+] Clean text chunks (ChromaDB target): {len(all_text_chunks)}")
print(f"[+] Isolated table chunks (Logged separately): {len(all_table_chunks)}")