import json

# Load the processed chunks
with open("D:/Courses/Programming/CRAG_PROJECT/data/processed/processed_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Filter by source_paper in metadata (or check if ID starts with the paper ID)
target_paper = "2305.18010v2"
paper_chunks = [c for c in chunks if c["metadata"].get("source_paper") == target_paper]

print(f"[+] Found {len(paper_chunks)} chunks for paper {target_paper}")

# Save to a separate JSON file if needed
with open(f"./data/{target_paper}_chunks.json", "w", encoding="utf-8") as f:
    json.dump(paper_chunks, f, indent=2, ensure_ascii=False)