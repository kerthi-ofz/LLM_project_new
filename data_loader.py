import os
from llama_index.core import SimpleDirectoryReader


def load_documents(data_dir: str = "./data"):
    """
    Load all documents from the given directory.
    Supports: PDF, .txt, .md, .docx and more.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory '{data_dir}' does not exist. "
            "Please create it and add your documents."
        )

    files = os.listdir(data_dir)
    if not files:
        raise ValueError(
            f"No files found in '{data_dir}'. "
            "Please add at least one PDF or text file."
        )

    print(f"[DataLoader] Loading documents from: {data_dir}")
    reader = SimpleDirectoryReader(input_dir=data_dir, recursive=True)
    documents = reader.load_data()
    print(f"[DataLoader] Loaded {len(documents)} document(s).")
    return documents