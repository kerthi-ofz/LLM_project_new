import argparse
from data_loader import load_documents
from vector_db import ingest_documents


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG vector database."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Path to the folder containing your documents (default: ./data)",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  Production RAG — Document Ingestion")
    print("=" * 50)

    documents = load_documents(data_dir=args.data_dir)
    index = ingest_documents(documents)

    print("\n✅ Done! Now run:")
    print("   streamlit run streamlit_app.py\n")


if __name__ == "__main__":
    main()