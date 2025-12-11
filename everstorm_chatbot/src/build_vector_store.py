from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings

def build_vector_store(chunks):
    """Build and save FAISS vector store from document chunks."""
    embedding_vector = []

    # Embed the sentence "Hello world! and store it in an embedding_vector.
    embeddings = SentenceTransformerEmbeddings(model_name="thenlper/gte-small")
    embedding_vector = embeddings.embed_query("Hello world!")
    print(len(embedding_vector))

    # Expected steps:
        # 1. Build the FAISS index from the list of document chunks and their embeddings.
        # 2. Create a retriever object with a suitable k value (e.g., 8).
        # 3. Save the vector store locally (e.g., under "faiss_index").
        # 4. Print a short confirmation showing how many embeddings were stored.

    vectordb = FAISS.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})
    vectordb.save_local("faiss_index")

    print("✅ Vector store with", vectordb.index.ntotal, "embeddings")
    
    return retriever, embeddings
