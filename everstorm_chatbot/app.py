import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Page config
st.set_page_config(page_title="Everstorm Support Chat", page_icon="🛍️")
st.title("🛍️ Everstorm Outfitters Support")
st.caption("Ask me anything about our policies, shipping, or returns!")

# System prompt
SYSTEM_TEMPLATE = """
You are a **Customer Support Chatbot**. Use only the information in CONTEXT to answer.
If the answer is not in CONTEXT, respond with "I'm not sure from the docs."

Rules:
1) Use ONLY the provided <context> to answer.
2) If the answer is not in the context, say: "I don't know based on the retrieved documents."
3) Be concise and accurate. Prefer quoting key phrases from the context.
4) When possible, cite sources as [source: source] using the metadata.

CONTEXT:
{context}

USER:
{question}
"""

# Load RAG components (cached to avoid reloading)
@st.cache_resource
def load_rag_chain():
    # Load embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="thenlper/gte-small")

    # Load FAISS index
    vectordb = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})

    # Initialize LLM
    llm = Ollama(model="gemma3:1b", temperature=0.1)

    # Create prompt and chain
    prompt = PromptTemplate(template=SYSTEM_TEMPLATE, input_variables=["context", "question"])
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return chain

# Initialize chain
chain = load_rag_chain()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_history = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about our refund policy, shipping times, or support hours..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chain({"question": prompt, "chat_history": st.session_state.chat_history})
            answer = result["answer"]
            st.markdown(answer)

    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.append((prompt, answer))

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.info("This chatbot answers questions about Everstorm Outfitters using RAG (Retrieval-Augmented Generation).")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.caption("Powered by Gemma 3 (1B) via Ollama")
