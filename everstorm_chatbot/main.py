# Import standard libraries for file handling and text processing
import os, pathlib, textwrap, glob

import os, pathlib, textwrap, glob
from langchain_community.document_loaders import UnstructuredURLLoader, TextLoader, PyPDFLoader

# Load documents from various sources (URLs, text files, PDFs)
from langchain_community.document_loaders import UnstructuredURLLoader, TextLoader, PyPDFLoader

# Split long texts into smaller, manageable chunks for embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store to store and retrieve embeddings efficiently using FAISS
from langchain.vectorstores import FAISS

# Generate text embeddings using OpenAI or Hugging Face models
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, SentenceTransformerEmbeddings

# Use local LLMs (e.g., via Ollama) for response generation
from langchain.llms import Ollama

# Build a retrieval chain that combines a retriever, a prompt, and an LLM
from langchain.chains import ConversationalRetrievalChain

# Create prompts for the RAG system
from langchain.prompts import PromptTemplate

# Import functions from separate modules
from src.load_documents import load_documents
from src.chunk_text import chunk_documents
from src.build_vector_store import build_vector_store
from src.test_llm import test_llm
from src.build_rag_chain import build_rag_chain
from src.test_rag import test_rag_chain

print("✅ Libraries imported! You're good to go!")

# 2 - Data preparation
print("\n=== Step 2: Data Preparation ===")
raw_docs = load_documents()

# 2.2 - Chunk the text
print("\n=== Step 2.2: Chunking Text ===")
chunks = chunk_documents(raw_docs)

# 3 - Build a retriever
print("\n=== Step 3: Build Retriever ===")
retriever, embeddings = build_vector_store(chunks)

# 4 - Build the generation engine
print("\n=== Step 4: Test LLM ===")
llm = test_llm()

# 5 - Build a RAG
print("\n=== Step 5: Build RAG Chain ===")
chain = build_rag_chain(retriever)

# 5.3 - Validate the RAG chain
print("\n=== Step 5.3: Test RAG Chain ===")
test_rag_chain(chain)

print("\n✅ RAG pipeline complete! All functions executed successfully.")
