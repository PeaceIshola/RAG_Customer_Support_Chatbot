#!/bin/bash

# RAG Chatbot Setup Script
# This script automates the setup process for the RAG Customer Support Chatbot

echo "🚀 Starting RAG Chatbot Setup..."
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Environment created"
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found. Please install Ollama:"
    echo ""
    echo "  macOS:   brew install ollama"
    echo "  Linux:   curl -fsSL https://ollama.com/install.sh | sh"
    echo "  Windows: Download from https://ollama.com/download"
    echo ""
    exit 1
fi

echo "✅ Ollama found"
echo ""

# Check if Ollama server is running
if ! curl -s http://localhost:11434 &> /dev/null; then
    echo "⚠️  Ollama server is not running."
    echo "Please start it in a separate terminal:"
    echo "  ollama serve"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✅ Ollama server is running"
echo ""

# Pull Gemma model
echo "📥 Pulling Gemma 3 1B model..."
ollama pull gemma3:1b

echo ""
echo "✅ Setup complete!"
echo ""
echo "To use the chatbot:"
echo "  1. Activate the environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run the pipeline (builds index):"
echo "     python main.py"
echo ""
echo "  3. Or run the web interface:"
echo "     streamlit run app.py"
echo ""
