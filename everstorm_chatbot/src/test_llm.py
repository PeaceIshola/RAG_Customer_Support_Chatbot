from langchain.llms import Ollama

def test_llm():
    """Test LLM with a random prompt (Sanity check)."""
    # Expected steps:
        # 1. Initialize the model (for example, gemma3:1b) with a low temperature such as 0.1 for more factual outputs.
        # 2. Use llm.invoke() with a short test prompt and print the response to verify that the model runs successfully.

    llm = Ollama(model="gemma3:1b", temperature=0.1)
    print(llm.invoke("What is the capital of France?"))
    return llm
