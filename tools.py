import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from crewai.tools import tool

DB_DIR = "./chroma_sop_db"
vectorstore = None

def get_embeddings():
    """Returns a fast, CPU-friendly embedding model."""
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def build_vectorstore_from_files(uploaded_files) -> int:
    """Parses uploaded SOP files, chunks text, and stores vectors in ChromaDB."""
    global vectorstore
    documents = []
    processed_count = 0

    for uploaded_file in uploaded_files:
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
                os.remove(tmp_path)
                continue

            docs = loader.load()
            for doc in docs:
                doc.metadata["source_name"] = uploaded_file.name
            documents.extend(docs)
            processed_count += 1

        except Exception as e:
            st.error(f"Error parsing `{uploaded_file.name}`: {str(e)}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if not documents:
        return 0

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    return processed_count

@tool("SOP Vector Search Tool")
def search_sop_database(query: str) -> str:
    """Search indexed SOP documents for relevant sections matching the user query."""
    global vectorstore
    if vectorstore is None:
        if os.path.exists(DB_DIR):
            vectorstore = Chroma(
                persist_directory=DB_DIR,
                embedding_function=get_embeddings()
            )
        else:
            return "No SOP database found. Please index files first."

    docs = vectorstore.similarity_search(query, k=4)
    if not docs:
        return "No relevant SOP sections found for this query."

    results = []
    for d in docs:
        source = d.metadata.get("source_name", "Unknown Source")
        results.append(f"--- Document: {source} ---\n{d.page_content}")

    return "\n\n".join(results)
