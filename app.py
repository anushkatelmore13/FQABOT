import os
import streamlit as st
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from src.extract import extract_text_from_file
from src.chunk import chunk_text
from src.embed import EmbeddingManager
from src.retrieve import retrieve_top_k
from src.generate import generate_answer

st.set_page_config(
    page_title="Document FAQ Assistant",
    page_icon="📚",
    layout="wide"
)

if "embedding_manager" not in st.session_state:
    st.session_state.embedding_manager = None

if "document_data" not in st.session_state:
    st.session_state.document_data = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------- SIDEBAR ----------------
st.sidebar.title("Document FAQ Assistant")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload Document",
    type=["pdf", "txt", "md", "markdown", "csv"],
    help="Supported formats: PDF, TXT, Markdown, and CSV."
)

st.sidebar.markdown("### Retrieval Settings")
threshold = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=float(os.getenv("SIMILARITY_THRESHOLD", 0.25)),
    step=0.05
)

top_k = st.sidebar.slider(
    "Top-K Chunks to Retrieve",
    min_value=1,
    max_value=10,
    value=int(os.getenv("TOP_K", 3)),
    step=1
)

st.sidebar.markdown("### LLM API Settings")
provider = st.sidebar.selectbox(
    "LLM Provider",
    options=["gemini", "openai"],
    index=0,
    help="Use a real Gemini or OpenAI key for answers."
)

api_key_input = st.sidebar.text_input(
    "API Key",
    type="password",
    help="You can also set GEMINI_API_KEY or OPENAI_API_KEY in a .env file."
)

if api_key_input:
    if provider == "openai":
        os.environ["OPENAI_API_KEY"] = api_key_input
    else:
        os.environ["GEMINI_API_KEY"] = api_key_input

provider_key_name = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
api_key_configured = bool(os.getenv(provider_key_name) or (provider == "gemini" and os.getenv("GOOGLE_API_KEY")))

if api_key_configured:
    st.sidebar.success(f"{provider.upper()} key configured")
else:
    st.sidebar.warning(f"Add a {provider.upper()} key before asking questions.")

st.sidebar.markdown("---")
if st.sidebar.button("Clear Session & History", use_container_width=True):
    st.session_state.document_data = None
    st.session_state.chat_history = []
    st.rerun()

# ---------------- CORE LOGIC ----------------
if uploaded_file is not None:
    current_filename = uploaded_file.name
    if (
        st.session_state.document_data is None or
        st.session_state.document_data["filename"] != current_filename
    ):
        with st.spinner(f"Extracting & Indexing '{current_filename}'..."):
            try:
                raw_text = extract_text_from_file(uploaded_file, current_filename)
                chunk_size = int(os.getenv("CHUNK_SIZE", 250))
                overlap = int(os.getenv("CHUNK_OVERLAP", 50))
                chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
                
                if st.session_state.embedding_manager is None:
                    st.session_state.embedding_manager = EmbeddingManager()
                
                embeddings = st.session_state.embedding_manager.fit_and_embed_chunks(chunks)

                st.session_state.document_data = {
                    "filename": current_filename,
                    "text": raw_text,
                    "chunks": chunks,
                    "embeddings": embeddings
                }
                st.session_state.chat_history = []
                st.toast(f"Successfully indexed {len(chunks)} chunks!")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# ---------------- MAIN DASHBOARD ----------------
st.title("Document FAQ Assistant")
st.caption("Upload a real PDF, TXT, Markdown, or CSV file, then ask grounded questions against the indexed content.")

doc = st.session_state.document_data

if doc is None:
    st.info("Upload a document in the sidebar to begin indexing.")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Indexed Document", doc["filename"])
    col2.metric("Total Character Count", f"{len(doc['text']):,}")
    col3.metric("Indexed Chunks", f"{len(doc['chunks'])}")
    
    st.markdown("---")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "retrieved_results" in message and message["retrieved_results"]:
                with st.expander("View Retrieved Source Chunks"):
                    for item in message["retrieved_results"]:
                        c = item["chunk"]
                        score = item["score"]
                        st.markdown(f"**Chunk #{c['id']}** | *Similarity Score: `{score:.4f}`*")
                        st.caption(c["text"])
                        st.divider()

    if user_question := st.chat_input("Ask a question about the document..."):
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context & generating grounded answer..."):
                query_vec = st.session_state.embedding_manager.embed_query(user_question)

                retrieval = retrieve_top_k(
                    query_vec=query_vec,
                    chunk_vecs=doc["embeddings"],
                    chunks=doc["chunks"],
                    top_k=top_k,
                    threshold=threshold
                )

                if api_key_configured:
                    answer = generate_answer(
                        query=user_question,
                        retrieved_results=retrieval["results"],
                        is_relevant=retrieval["is_relevant"],
                        provider=provider
                    )
                else:
                    answer = (
                        f"Add a real {provider.upper()} API key in the sidebar or .env file, "
                        "then ask again. The app does not use mock answers."
                    )

                st.markdown(answer)

                if retrieval["results"]:
                    with st.expander("View Retrieved Source Chunks"):
                        for item in retrieval["results"]:
                            c = item["chunk"]
                            score = item["score"]
                            st.markdown(f"**Chunk #{c['id']}** | *Similarity Score: `{score:.4f}`*")
                            st.caption(c["text"])
                            st.divider()

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "retrieved_results": retrieval["results"]
        })
