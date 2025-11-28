# src/nfl_assistant.py
import os
import argparse

from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INDEX_DIR = "index/chroma"
COLLECTION_NAME = "nfl_knowledge"


def get_chroma_collection():
    chroma_client = chromadb.PersistentClient(
        path=INDEX_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    return chroma_client.get_collection(COLLECTION_NAME)


def retrieve_context(query: str, k: int = 5) -> list[str]:
    collection = get_chroma_collection()

    # Embed the query
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query],
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[emb],
        n_results=k,
    )
    documents = results.get("documents", [[]])[0]
    return documents


# ---------- Prompts ----------

def build_rule_explain_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return f"""
You are an assistant that explains NFL rules and analytics using ONLY the provided context.

Context (excerpts from the NFL rulebook and analytics notes):
{context}

Question: {question}

Instructions:
- Answer in clear, casual language.
- Base your answer ONLY on the context above. If the answer is not in the context, say you don't know.
- If relevant, quote or reference the rule or section in plain English.
- Keep the answer under 250 words.

Now answer the question.
"""


def build_rule_compare_prompt(question: str, context_chunks: list[str]) -> str:
    """
    Prompt for comparing two (or more) rules.
    The question will usually look like: "Compare defensive pass interference and offensive pass interference."
    """
    context = "\n\n---\n\n".join(context_chunks)
    return f"""
You are an assistant that compares NFL rules using ONLY the provided context.

Context (excerpts from the NFL rulebook and analytics notes):
{context}

Question: {question}

Instructions:
- Identify the two main rules being compared from the question (for example, "Defensive Pass Interference (DPI)" vs "Offensive Pass Interference (OPI)").
- First, briefly define each rule separately.
- Then, compare them directly with a focus on:
  - When they apply
  - Yardage / enforcement (spot foul vs previous spot, automatic first down, etc.)
  - Who commits the foul (offense or defense)
- Present the comparison as:
  - A short bullet list **or**
  - A simple text table-style layout (no markdown table needed, just clear formatting).
- Base your answer ONLY on the context above. If you cannot fully answer from the context, say what is missing.
- Keep the answer under 300 words.

Now answer the question.
"""


# ---------- Answer helpers ----------

def answer_rule_question(question: str) -> str:
    context_chunks = retrieve_context(question, k=5)
    prompt = build_rule_explain_prompt(question, context_chunks)

    resp = client.responses.create(
        model="gpt-5.1-mini",
        input=prompt,
    )
    return resp.output[0].content[0].text


def answer_rule_compare(question: str) -> str:
    context_chunks = retrieve_context(question, k=6)
    prompt = build_rule_compare_prompt(question, context_chunks)

    resp = client.responses.create(
        model="gpt-5.1-mini",
        input=prompt,
    )
    return resp.output[0].content[0].text


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="NFL Rules & Analytics RAG Assistant")
    parser.add_argument(
        "mode",
        choices=["rule-explain", "rule-compare"],
        help="Mode to run.",
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="+",
        help="Question to ask the assistant.",
    )
    args = parser.parse_args()

    question = " ".join(args.question)

    if args.mode == "rule-explain":
        print(f"🟦 Question: {question}\n")
        answer = answer_rule_question(question)
        print("🟩 Answer:\n")
        print(answer)

    elif args.mode == "rule-compare":
        print(f"🟦 Compare request: {question}\n")
        answer = answer_rule_compare(question)
        print("🟩 Comparison:\n")
        print(answer)


if __name__ == "__main__":
    main()
