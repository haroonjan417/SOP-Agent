import os
import tempfile
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

# Global or persistent Chroma DB path
DB_DIR = "./chroma_sop_db"
vectorstore = None

def get_embeddings():
    """Initializes lightweight, local CPU embeddings without external API costs."""
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def build_vectorstore_from_files(uploaded_files) -> int:
    """Processes uploaded PDF and TXT files, chunks text, and stores vectors in ChromaDB."""
    global vectorstore
    documents = []

    for uploaded_file in uploaded_files:
        # Create a temporary file to save uploaded content
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif file_extension == ".txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
            else:
                continue

            docs = loader.load()
            # Attach source metadata to each document chunk
            for doc in docs:
                doc.metadata["source_name"] = uploaded_file.name
            documents.extend(docs)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if not documents:
        return 0

    # Split documents into optimal chunks for RAG
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    # Initialize Chroma vector store with chunked documents
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    return len(uploaded_files)

def query_sop_vectorstore(query: str, k: int = 3) -> str:
    """Retrieves relevant SOP chunks from ChromaDB formatted with exact document citations."""
    global vectorstore

    # Load existing vector database if not in memory
    if vectorstore is None:
        if os.path.exists(DB_DIR):
            embeddings = get_embeddings()
            vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        else:
            return "No SOP documents indexed in the vector store yet. Please upload SOPs first."

    # Perform similarity search
    results = vectorstore.similarity_search(query, k=k)
    
    if not results:
        return "No relevant SOP procedures found matching this incident."

    formatted_context = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source_name", "Uploaded_SOP_Document")
        formatted_context.append(f"--- SOP Chunk {i} [Document Source: {source}] ---\n{doc.page_content}")

    return "\n\n".join(formatted_context)
