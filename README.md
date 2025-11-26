# nfl-rules-rag-assistant 🏈

**nfl-rules-rag-assistant** is a small GenAI project that uses a
**Retrieval-Augmented Generation (RAG)** pipeline to explain NFL rules
(and eventually analytics concepts) in plain English.

It:

- Ingests text from the **NFL rulebook** and your own **analytics notes**
- Stores it in a local **vector database**
- Uses OpenAI embeddings + an LLM to:
  - Retrieve the most relevant rule/analytics snippets
  - Answer questions like:

    - “What is roughing the passer?”
    - “What’s the difference between illegal contact and defensive holding?”
    - “What actually counts as a catch in the NFL?”

All answers are grounded in your local text, not the model’s memory.
