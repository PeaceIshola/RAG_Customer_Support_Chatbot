from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

def build_rag_chain(retriever):
    """Create a RAG chain by combining retriever, prompt, and LLM."""
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

    # Expected steps:
        # 1. Create a PromptTemplate that uses the SYSTEM_TEMPLATE you defined earlier, with input variables for "context" and "question".
        # 2. Initialize your LLM using Ollama with the gemma3:1b model and a low temperature (e.g., 0.1) for reliable, grounded responses.
        # 3. Build a ConversationalRetrievalChain by combining the LLM, the retriever, and your custom prompt and name it "chain".

    prompt = PromptTemplate(template=SYSTEM_TEMPLATE, input_variables=["context", "question"])
    llm = Ollama(model="gemma3:1b", temperature=0.1)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, combine_docs_chain_kwargs={"prompt": prompt})
    
    return chain
