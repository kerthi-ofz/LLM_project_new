import streamlit as st
from vector_db import load_index, get_query_engine

st.set_page_config(
    page_title="Production RAG Chat",
    page_icon="📚",
    layout="wide",
)
st.title("📚 Production RAG — Ask Your Documents")
st.caption("Powered by LlamaIndex + Qdrant + Ollama (100% local, no API key needed)")


@st.cache_resource(show_spinner="Loading index from Qdrant...")
def get_engine():
    """Load the query engine once and cache it."""
    try:
        index = load_index()
        return get_query_engine(index, similarity_top_k=3)
    except Exception as e:
        return None


engine = get_engine()

if engine is None:
    st.error("Please run python main.py first to ingest your documents.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = engine.query(prompt)
                answer = str(response)
                st.markdown(answer)

                if hasattr(response, "source_nodes") and response.source_nodes:
                    with st.expander("📄 View Source Chunks"):
                        for i, node in enumerate(response.source_nodes):
                            st.markdown(f"*Chunk {i+1}* (score: {node.score:.4f})")
                            st.text(node.node.get_content()[:500] + "...")
                            st.divider()
            except Exception as e:
                answer = f"Error: {e}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.header("How it works")
    st.markdown("""
    1. Ingest docs → python main.py
    2. Question → embedded by Ollama
    3. Similar chunks → retrieved from Qdrant
    4. Ollama → generates answer from chunks
    """)
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()