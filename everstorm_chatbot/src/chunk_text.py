from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(raw_docs):
    """Split documents into smaller chunks for embedding."""
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = text_splitter.split_documents(raw_docs)
    print(f"✅ {len(chunks)} chunks ready for embedding")
    return chunks
