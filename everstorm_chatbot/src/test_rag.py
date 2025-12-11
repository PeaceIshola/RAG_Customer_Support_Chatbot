def test_rag_chain(chain):
    """Validate the RAG chain with test questions."""
    test_questions = [
        "If I'm not happy with my purchase, what is your refund policy and how do I start a return?",
        "How long will delivery take for a standard order, and where can I track my package once it ships?",
        "What's the quickest way to contact your support team, and what are your operating hours?",
    ]

    # Expected steps:
        # 1. Initialize an empty chat_history list.
        # 2. Loop through test_questions, pass each question and the current chat history to the chain, and append the new answer.
        # 3. Print each question and the LLM's response to verify it's working correctly.

    chat_history = []
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        result = chain({"question": question, "chat_history": chat_history})
        answer = result["answer"]
        print(f"💬 Answer: {answer}")
        chat_history.append((question, answer))
