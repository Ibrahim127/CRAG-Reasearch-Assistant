import os
import json
import time
import requests
import urllib.parse
import xml.etree.ElementTree as ET
from src.config import VECTORSTORE_PATH, ARXIV_QUERY, FOUNDATIONAL_COUNT, RECENT_COUNT

RAW_PDFS_DIR = "data/raw_pdfs"
MANIFEST_PATH = os.path.join(RAW_PDFS_DIR, "manifest.json")

# Ensure directories exist
os.makedirs(RAW_PDFS_DIR, exist_ok=True)

def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_manifest(manifest_data):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4, ensure_ascii=False)

def fetch_arxiv_metadata(query, max_results, sort_by):
    """Queries the arXiv API and parses matching entries into a metadata dictionary."""
    encoded_query = urllib.parse.quote(query)
    url = f"https://export.arxiv.org/api/query?search_query={encoded_query}&max_results={max_results}&sortBy={sort_by}&sortOrder=descending"
    
    print(f"Querying arXiv API ({sort_by})...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    # Atom XML namespaces
    ns = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }
    
    root = ET.fromstring(response.content)
    papers = {}
    
    for entry in root.findall('atom:entry', ns):
        # Extract ID (e.g., http://arxiv.org/abs/2401.15884v1 -> 2401.15884v1)
        raw_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
        
        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
        published = entry.find('atom:published', ns).text
        
        authors = [auth.find('atom:name', ns).text for auth in entry.findall('atom:author', ns)]
        categories = [cat.get('term') for cat in entry.findall('atom:category', ns)]
        
        # Look for the direct PDF link
        pdf_url = ""
        for link in entry.findall('atom:link', ns):
            if link.get('title') == 'pdf' or link.get('type') == 'application/pdf':
                pdf_url = link.get('href')
                break
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{raw_id}.pdf"
            
        papers[raw_id] = {
            "title": title,
            "authors": authors,
            "published": published,
            "categories": categories,
            "pdf_url": pdf_url,
            "abstract": summary
        }
    return papers

def download_paper_pdf(arxiv_id, pdf_url):
    """Downloads a single PDF to data/raw_pdfs/."""
    safe_id = arxiv_id.replace("/", "_")
    pdf_path = os.path.join(RAW_PDFS_DIR, f"{safe_id}.pdf")
    
    if os.path.exists(pdf_path):
        return pdf_path # Skip if already downloaded
        
    try:
        res = requests.get(pdf_url, timeout=30)
        res.raise_for_status()
        with open(pdf_path, "wb") as f:
            f.write(res.content)
        return pdf_path
    except Exception as e:
        print(f"Skipping PDF download for {arxiv_id} due to error: {e}")
        return None

def pipeline():
    manifest = load_manifest()
    
    # 1. Fetch Foundational Papers (Sorted by Relevance)
    foundational = fetch_arxiv_metadata(ARXIV_QUERY, FOUNDATIONAL_COUNT, "relevance")
    #pause a bit for runtime
    print("Pausing for 5 seconds to respect arXiv API rate limits...")
    time.sleep(5)
    # 2. Fetch Recent Papers (Sorted by Submitted Date)
    recent = fetch_arxiv_metadata(ARXIV_QUERY, RECENT_COUNT, "submittedDate")
    
    # Merge targets
    all_targets = {**foundational, **recent}
    print(f"Found total of {len(all_targets)} unique candidate papers.\n")
    
    # Download loop
    for idx, (arxiv_id, meta) in enumerate(all_targets.items(), 1):
        if arxiv_id in manifest:
            print(f"[{idx}/{len(all_targets)}] {arxiv_id} already exists in manifest. Skipping.")
            continue
            
        print(f"[{idx}/{len(all_targets)}] Processing {arxiv_id} - {meta['title'][:50]}...")
        local_path = download_paper_pdf(arxiv_id, meta['pdf_url'])
        
        if local_path:
            # Add mapping catalog details exactly matching your schema specification
            manifest[arxiv_id] = {
                "arxiv_id": arxiv_id,
                "title": meta["title"],
                "authors": meta["authors"],
                "published": meta["published"],
                "categories": meta["categories"],
                "pdf_url": meta["pdf_url"],
                "abstract": meta["abstract"],
                "local_path": local_path
            }
            save_manifest(manifest)
            # Polite backoff delay to respect arXiv's rate limits
            time.sleep(2)

    print("\nInitialization data pipeline run completed successfully!")

if __name__ == "__main__":
    pipeline()