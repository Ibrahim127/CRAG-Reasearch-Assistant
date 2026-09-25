
import sys
import tempfile
import urllib.request
import re
import warnings
from pathlib import Path

import streamlit as st
import arxiv
import chromadb

warnings.filterwarnings("ignore", category=PendingDeprecationWarning, module="langgraph")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.build_graph import app as crag_app
from src.graph import config
from src.ingestion.ingest_documents import ingest_pdf_folder


def get_existing_collections(chroma_path: str) -> list[str]:
    try:
        chromadb.api.client.SharedSystemClient.clear_system_cache() 
        
        client = chromadb.PersistentClient(path=chroma_path)
        collections = client.list_collections()
        return [c.name for c in collections]
    except Exception as e:
        print(f"Error fetching collections: {e}")
        return []


def validate_collection_name(name: str) -> tuple[bool, str]:
    
    if not name or not name.strip():
        return False, "Collection name cannot be empty."
    
    cleaned = name.strip()
    
    if len(cleaned) < 3:
        return False, "Name must be at least 3 characters long."
    if len(cleaned) > 63:
        return False, "Name cannot exceed 63 characters."
    if not re.match(r"^[a-zA-Z0-9]", cleaned):
        return False, "Name must start with a letter or number."
    if not re.match(r".*[a-zA-Z0-9]$", cleaned):
        return False, "Name must end with a letter or number."
    if not re.match(r"^[a-zA-Z0-9._-]+$", cleaned):
        return False, "Only letters, numbers, hyphens (-), underscores (_), and periods (.) are allowed."
    if ".." in cleaned:
        return False, "Name cannot contain consecutive periods ('..')."
    if re.match(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$", cleaned):
        return False, "Name cannot be an IP address."
        
    return True, ""


def fetch_arxiv_papers(query: str, max_results: int, output_dir: Path) -> list[str]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    downloaded_files = []
    for result in client.results(search):
        safe_title = "".join([c for c in result.title if c.isalnum() or c.isspace()]).rstrip()[:40]
        filename = f"{result.get_short_id()}_{safe_title}.pdf".replace(" ", "_")
        pdf_path = output_dir / filename
        
        try:
            urllib.request.urlretrieve(result.pdf_url, str(pdf_path))
            if pdf_path.exists() and pdf_path.stat().st_size > 0:
                downloaded_files.append(str(pdf_path))
        except Exception as e:
            print(f"Failed to download {result.title}: {e}")
            
    return downloaded_files


def run_app():
    st.set_page_config(page_title="CRAG Research Assistant", layout="wide")
    st.title("CRAG Research Assistant")
    st.caption("Corrective RAG over your own documents -- upload PDFs or fetch from arXiv.")

    if "active_collection" not in st.session_state:
        st.session_state.active_collection = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar: upload or fetch + ingest
    with st.sidebar:
        st.header("1. Add documents")

        tab_upload, tab_arxiv = st.tabs(["📁 Upload PDFs", "🔍 Search arXiv"])

        with tab_upload:
            collection_name_upload = st.text_input(
                "New Collection name",
                placeholder="e.g. my_research_papers",
                help="3-63 chars, alphanumeric, hyphens, periods, underscores. Must start/end with a letter or number.",
                key="coll_upload"
            )

            is_valid_upload, error_msg_upload = validate_collection_name(collection_name_upload)
            
            if collection_name_upload and not is_valid_upload:
                st.caption(f" :red[{error_msg_upload}]")

            uploaded_files = st.file_uploader(
                "Upload PDF files", type=["pdf"], accept_multiple_files=True
            )

            ingest_upload_clicked = st.button(
                "Ingest Uploaded PDFs",
                disabled=not (uploaded_files and is_valid_upload),
                use_container_width=True,
                key="btn_upload"
            )

            if ingest_upload_clicked:
                final_coll_name = collection_name_upload.strip()
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    for f in uploaded_files:
                        pdf_file_path = tmp_path / f.name
                        pdf_file_path.write_bytes(f.getvalue())

                    with st.spinner(f"Extracting, chunking, and embedding {len(uploaded_files)} PDF(s)..."):
                        try:
                            ui_bar = st.progress(0, text="Starting extraction...")
                            result = ingest_pdf_folder(
                                pdf_dir=str(tmp_path),
                                collection_name=final_coll_name,
                                chroma_path=config.CHROMA_PATH,
                                append=False,
                                progress_callback=lambda p, text: ui_bar.progress(p, text=text)
                            )
                            ui_bar.empty()
                            st.session_state.active_collection = result["collection_name"]
                            st.session_state.chat_history = []
                            st.success(f"Ingested {result['total_documents']} document(s) into '{result['collection_name']}'.")
                        except Exception as e:
                            st.error(f"Ingestion failed: {e}")

        with tab_arxiv:
            arxiv_query = st.text_input("arXiv Query", placeholder="e.g. Vision Language Models", key="query_arxiv")
            max_papers_arxiv = st.number_input("Number of papers to pull", min_value=1, max_value=25, value=5, key="num_arxiv")
            
            collection_name_arxiv = st.text_input(
                "New Collection name", 
                placeholder="e.g. arxiv_vlm", 
                help="3-63 chars, alphanumeric, hyphens, periods, underscores.",
                key="coll_arxiv"
            )
            
            is_valid_arxiv, error_msg_arxiv = validate_collection_name(collection_name_arxiv)
            
            if collection_name_arxiv and not is_valid_arxiv:
                st.caption(f" :red[{error_msg_arxiv}]")

            ingest_arxiv_clicked = st.button(
                "Fetch & Ingest", 
                disabled=not (arxiv_query and is_valid_arxiv), 
                key="btn_arxiv", 
                use_container_width=True
            )

            if ingest_arxiv_clicked:
                final_coll_name = collection_name_arxiv.strip()
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    with st.spinner(f"Fetching papers for '{arxiv_query}' from arXiv..."):
                        downloaded_files = fetch_arxiv_papers(arxiv_query, max_papers_arxiv, tmp_path)
                        if not downloaded_files:
                            st.warning("No valid PDFs found.")
                        else:
                            ui_bar = st.progress(0, text="Starting extraction...")
                            result = ingest_pdf_folder(
                                pdf_dir=str(tmp_path), collection_name=final_coll_name, chroma_path=config.CHROMA_PATH,
                                append=False, progress_callback=lambda p, text: ui_bar.progress(p, text=text)
                            )
                            ui_bar.empty()
                            st.session_state.active_collection = result["collection_name"]
                            st.session_state.chat_history = []
                            st.success(f"Ingested {result['total_documents']} paper(s) into '{result['collection_name']}'.")

        st.divider()
        st.header("2. Choose a collection")

        existing_collections = get_existing_collections(config.CHROMA_PATH)
        
        if existing_collections:
            default_index = 0
            if st.session_state.active_collection in existing_collections:
                default_index = existing_collections.index(st.session_state.active_collection)
            elif config.COLLECTION_NAME in existing_collections:
                default_index = existing_collections.index(config.COLLECTION_NAME)
                
            selected_collection = st.selectbox("Select a database", options=existing_collections, index=default_index)
            if st.button("Use this collection", use_container_width=True):
                st.session_state.active_collection = selected_collection
                st.session_state.chat_history = []
        else:
            st.info("No collections found in the database. Please add some documents first.")

        if st.session_state.active_collection:
            st.info(f"Active: **{st.session_state.active_collection}**")

    # Main: chat interface
    if not st.session_state.active_collection:
        st.info("Upload documents, fetch from arXiv, or select a collection in the sidebar to get started.")
        st.stop()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                with st.expander("Routing details"):
                    st.write(f"**Routing decision:** `{msg['meta']['retrieval_action']}`")
                    st.write(f"**Grounded:** `{msg['meta']['hallucination_status']}`")
                    st.write(f"**Context sources used:** {msg['meta']['context_items']}")

    question = st.chat_input("Ask a question about your documents...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving, grading, and generating..."):
                initial_state = {
                    "question": question,
                    "collection_name": st.session_state.active_collection,
                    "rewritten_query": None,
                    "retrieved_docs": [],
                    "web_docs": [],
                    "retrieval_action": None,
                    "final_context": [],
                    "generation": None,
                    "hallucination_status": None,
                    "retry_count": 0,
                }
                final_state = crag_app.invoke(initial_state)

            answer = final_state.get("generation") or "No answer was generated."
            st.markdown(answer)

            meta = {
                "retrieval_action": final_state.get("retrieval_action"),
                "hallucination_status": final_state.get("hallucination_status"),
                "context_items": len(final_state.get("final_context", [])),
            }
            with st.expander("Routing details"):
                st.write(f"**Routing decision:** `{meta['retrieval_action']}`")
                st.write(f"**Grounded:** `{meta['hallucination_status']}`")
                st.write(f"**Context sources used:** {meta['context_items']}")

        st.session_state.chat_history.append({"role": "assistant", "content": answer, "meta": meta})

if __name__ == "__main__":
    run_app()