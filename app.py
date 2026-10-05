"""
app.py
------
Streamlit demo. Fully offline: image vectors come from your own
trained CNN, text matching + answer generation run locally too.

Run with:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st

from retrieval import VisionRetriever
from generate_answer import generate_answer

st.set_page_config(page_title="Vision RAG Assistant (Offline)", layout="wide")
st.title("🔍 Retrieval-Augmented Vision AI Assistant")
st.sidebar.success("Running fully offline — no API calls, no internet required.")
st.sidebar.markdown(
    "**Image vectors:** your own trained CNN (`embedding_model.pth`)\n\n"
    "**Text matching:** local MiniLM\n\n"
    "**Answer generation:** local TinyLlama-1.1B-Chat"
)

INDEX_DIR = "./index_store"

if not os.path.exists(INDEX_DIR):
    st.error(
        "No index found. First run:\n\n"
        "1. `python train_embedding_model.py`\n"
        "2. `python build_index.py --images_dir <your_folder>`"
    )
    st.stop()


@st.cache_resource
def load_retriever():
    return VisionRetriever(INDEX_DIR)


retriever = load_retriever()

mode = st.radio("Query type", ["Text query (keyword-style)", "Image query"], horizontal=True)
k = st.slider("Number of results to retrieve", min_value=1, max_value=10, value=5)

query_text = None
query_image_path = None

if mode.startswith("Text"):
    query_text = st.text_input("Type a word/phrase related to what you're looking for")
else:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
        tmp.write(uploaded.read())
        tmp.close()
        query_image_path = tmp.name
        st.image(query_image_path, caption="Query image", width=250)
    question_for_image = st.text_input(
        "What do you want to know about the similar images found?",
        value="What do these results have in common?",
    )

if st.button("Search & Answer"):
    with st.spinner("Retrieving relevant images..."):
        if mode.startswith("Text"):
            if not query_text:
                st.warning("Please enter a query.")
                st.stop()
            results = retriever.query_by_text(query_text, k=k)
            question = query_text
        else:
            if not query_image_path:
                st.warning("Please upload an image.")
                st.stop()
            results = retriever.query_by_image(query_image_path, k=k)
            question = question_for_image

    if not results:
        st.warning("No results found.")
        st.stop()

    st.subheader("Retrieved Results")
    cols = st.columns(min(len(results), 5))
    for i, r in enumerate(results):
        with cols[i % len(cols)]:
            st.image(r["path"], caption=f"{r['filename']}\npredicted: {r['caption']} (score={r['score']:.3f})")

    with st.spinner("Generating answer with local TinyLlama..."):
        answer = generate_answer(question, results)

    st.subheader("Assistant's Answer")
    st.write(answer)

    if os.path.exists(str(query_image_path or "")):
        try:
            os.remove(query_image_path)
        except OSError:
            pass
