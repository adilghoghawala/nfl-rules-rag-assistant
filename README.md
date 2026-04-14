# nfl-rules-rag-assistant 🏈

**nfl-rules-rag-assistant** is a small GenAI project that uses a  
**Retrieval-Augmented Generation (RAG)** pipeline to explain NFL rules  
(and eventually analytics concepts) in plain English.

---

## What it does

- Ingests text from the **NFL rulebook** and your own **analytics notes**
- Stores that knowledge in a local **vector database**
- Uses **OpenAI embeddings + an LLM** to:
  - Retrieve the most relevant rule / analytics snippets
  - Generate grounded explanations to questions like:
    - “What is roughing the passer?”
    - “What’s the difference between illegal contact and defensive holding?”
    - “What actually counts as a catch in the NFL?”

All answers are **grounded in your local text**, not just the model’s general memory.


---

## Features

- **RAG over NFL rules & notes**  
  - Load your own `rulebook.txt` and analytics text files
  - Chunk + embed them into a local ChromaDB index

- **LLM-powered rule explanations**  
  - Ask natural language questions like:
    - `What is defensive pass interference?`
    - `How is offensive pass interference enforced?`
  - The assistant uses only retrieved context to build answers

- **Grounded, low-hallucination design**  
  - Prompt explicitly tells the model to use **only** provided context
  - If something isn’t in your data, it’s allowed to say “I don’t know”

## Tech stack

- **Language:** Python
- **LLM:** OpenAI (chat + embedding APIs)
- **Vector store:** ChromaDB (persistent local index)
- **Config:** dotenv (`.env` for secrets)
- **Interface:** Command-line (CLI) for now

## How it works (architecture)

1. **Ingestion & chunking**
   - Load `.txt` files under `data/rulebook/` and `data/analytics/`.
   - Split them into smaller, paragraph-sized chunks for better retrieval.

2. **Embedding & indexing**
   - Use OpenAI’s `text-embedding-3-small` model to embed each chunk.
   - Store embeddings + raw text in a local ChromaDB collection at `index/chroma/`.

3. **Retrieval**
   - When you ask a question, embed the query and retrieve the top-k most similar chunks from the index.

4. **Grounded generation (RAG)**
   - Build a prompt that includes:
     - The retrieved context
     - The user’s question
     - Instructions to **only** use the provided context and say “I don’t know” if needed.
   - Call an OpenAI chat model (e.g. `gpt-5.1-mini`) to generate the final explanation.

### Compare two rules

Use the `rule-compare` mode to compare two penalties or rules side-by-side.

Examples:

```bash
py src/nfl_assistant.py rule-compare "Compare defensive pass interference and offensive pass interference"
py src/nfl_assistant.py rule-compare "Compare DPI and OPI"
py src/nfl_assistant.py rule-compare "Compare illegal contact vs defensive holding"
```

## Limitations

- The assistant only knows what you put under `data/`. If a rule or concept
  isn’t included there, it may honestly say it doesn’t know.
- The summaries are **not** official NFL legal wording; they are informal,
  plain-English explanations based on your text.
- Retrieval quality depends on how well the rulebook/notes are written and chunked.
- This project is intended for learning and personal use, not for making
  officiating decisions or betting decisions.


---

## 🗂 Project structure

```text
nfl-rules-rag-assistant/
  ├─ data/
  │   ├─ rulebook/
  │   │   └─ rulebook.txt          # your NFL rule summaries
  │   └─ analytics/
  │       └─ analytics_notes.txt   # optional: analytics articles/notes
  ├─ index/
  │   └─ chroma/                   # vector DB will be created here
  ├─ src/
  │   ├─ build_index.py            # ingests docs & builds vector index
  │   └─ nfl_assistant.py          # CLI for asking rule questions (RAG)
  ├─ .env.example
  ├─ requirements.txt
  ├─ .gitignore
  └─ README.md
