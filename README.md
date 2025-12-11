# RAG Customer Support Chatbot

A retrieval-augmented generation (RAG) chatbot that answers customer support questions using local LLMs and vector search.

## Features

- Modular Python architecture with separate files for each function
- FAISS vector store for efficient document retrieval
- Local LLM inference using Ollama (Gemma 3 1B)
- PDF and web page document ingestion
- Interactive Streamlit web interface

## What it does

- Ingests and chunks unstructured documents (PDFs and web pages)
- Creates embeddings and indexes them with FAISS
- Retrieves relevant context for user queries
- Generates responses using a local LLM (Gemma 3 1B)
- Provides a web interface for chatting with the bot

## Quick Setup

### Option 1: Automated Script (Recommended)

**Prerequisites:** 
- Python 3.8 or higher
- [Ollama](https://ollama.com/download)

```bash
cd everstorm_chatbot

# Start Ollama (in separate terminal, keep running)
ollama serve

# Run setup
./setup.sh

# Activate environment
source venv/bin/activate

# Run application
python main.py          # Build pipeline
streamlit run app.py    # Web interface
```

### Option 2: Manual Setup

#### Step 1: Check Python Version
Verify Python 3.8+:
```bash
python3 --version
```

If not installed, download from [python.org](https://www.python.org/downloads/)

#### Step 2: Create Virtual Environment
```bash
cd everstorm_chatbot
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- Python 3.11
- langchain (0.3.25)
- langchain-community (0.3.24)
- sentence-transformers (4.1.0)
- streamlit (1.45.1)
- faiss-cpu (1.11.0)
- unstructured (0.17.2)
- pypdf (5.1.0)

#### Step 3: Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:** Download from [ollama.com](https://ollama.com/download)

#### Step 4: Start Ollama Server
Open a terminal and keep this running:
```bash
ollama serve
```

#### Step 5: Pull the Model
In a new terminal:
```bash
ollama pull gemma3:1b
```

This downloads the Gemma 3 1B model (~1.5GB)

#### Step 6: Run the Application

**Build the pipeline:**
```bash
python main.py
```
This will:
1. Load PDF documents from `data/` folder
2. Chunk documents into smaller pieces
3. Generate embeddings and build FAISS index
4. Test the LLM connection
5. Validate the RAG chain with test questions

**Launch the web interface:**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

## Testing Individual Components

You can test each component separately for debugging or development:

### Test LLM Connection
```bash
python -c "from src.test_llm import test_llm; test_llm()"
```

### Load and Count Documents
```bash
python -c "from src.load_documents import load_documents; docs = load_documents(); print(f'Loaded {len(docs)} documents')"
```

### Create Document Chunks
```bash
python -c "from src.load_documents import load_documents; from src.chunk_text import chunk_documents; docs = load_documents(); chunks = chunk_documents(docs); print(f'Created {len(chunks)} chunks')"
```

### Build Vector Store Only
```bash
python -c "from src.load_documents import load_documents; from src.chunk_text import chunk_documents; from src.build_vector_store import build_vector_store; docs = load_documents(); chunks = chunk_documents(docs); build_vector_store(chunks)"
```

### Test RAG Chain (requires existing FAISS index)
```bash
python -c "from src.build_rag_chain import build_rag_chain; from src.test_rag import test_rag_chain; from langchain.vectorstores import FAISS; from langchain.embeddings import SentenceTransformerEmbeddings; embeddings = SentenceTransformerEmbeddings(model_name='thenlper/gte-small'); vectordb = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True); retriever = vectordb.as_retriever(search_kwargs={'k': 8}); chain = build_rag_chain(retriever); test_rag_chain(chain)"
```

## Directory Structure

```
everstorm_chatbot/
├── main.py                    # Main script that runs the full pipeline
├── app.py                     # Streamlit web interface
├── setup.sh                   # Automated setup script
├── Dockerfile                 # Docker container definition
├── docker-compose.yml         # Docker orchestration
├── environment.yml            # Conda environment specification
├── src/                       # Source code modules
│   ├── __init__.py           # Package initialization
│   ├── load_documents.py     # Loads PDF files and web pages
│   ├── chunk_text.py         # Splits documents into chunks
│   ├── build_vector_store.py # Builds FAISS vector store
│   ├── test_llm.py           # Tests the LLM connection
│   ├── build_rag_chain.py    # Creates the RAG chain
│   └── test_rag.py           # Validates RAG with test questions
├── data/                      # PDF documents (source data)
│   ├── Everstorm_Payment_refund_and_security.pdf
│   ├── Everstorm_Product_sizing_and_care_guide.pdf
│   ├── Everstorm_Return_and_exchange_policy.pdf
│   └── Everstorm_Shipping_and_Delivery_Policy.pdf
└── faiss_index/               # Vector store index (generated)
```

## Module Descriptions

- **src/build_vector_store.py** - Creates embeddings using `thenlper/gte-small` model and builds FAISS index
- **src/test_llm.py** - Tests Ollama connection with Gemma 3 1B model
- **src/build_rag_chain.py** - Creates the RAG chain combining retriever, prompt template, and LLM
- **src/test_rag.py** - Validates the RAG chain with predefined test questions
- **main.py** - Orchestrates all modules to build the complete pipeline
- **app.py** - Streamlit web interface for interactive chatting

## Technologies Used

- **LangChain** - Framework for building LLM applications
- **FAISS** - Vector store for efficient similarity search
- **Sentence Transformers** - Text embedding generation
- **Ollama** - Local LLM inference (Gemma 3 1B)
- **Streamlit** - Web interface framework

## Troubleshooting

### Virtual environment not activated
```bash
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### Module not found
Make sure virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Ollama not responding
```bash
# Check if running
curl http://localhost:11434

# Restart Ollama
pkill ollama
ollama serve
```

### FAISS index missing
```bash
# Rebuild index
python main.py
```

### Docker port conflict
Edit `docker-compose.yml` and change the port:
``**Add your own documents:** Place PDF files in `data/` folder
- **Change system prompt:** Edit `src/build_rag_chain.py`
- **Modify UI:** Edit `app.py`
- **Use different model:** Change model name in `src/test_llm.py` and `src/build_rag_chain.py`
- **Adjust chunk size:** Edit parameters in `src/chunk_text.py`
- **Change retrieval count:** Modify `k` value in `src/build_vector_store.py`

