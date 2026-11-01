
from pathlib import Path
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm


POLICY_DIR = Path("data/policies")
EVAL_PATH = Path("data/raw/policy_eval.csv")

CHUNK_SIZE = 40
OVERLAP = 10


# Part D, Task 1: Split policy files into overlapping chunks
def chunk_policy_file(path):
    text = path.read_text(encoding="utf-8")
    words = text.split()

    chunks = []
    step = CHUNK_SIZE - OVERLAP

    for start in range(0, len(words), step):
        chunk_words = words[start:start + CHUNK_SIZE]

        if not chunk_words:
            break

        chunks.append({
            "source": path.name,
            "text": " ".join(chunk_words),
        })

        # Stop when the current chunk reaches the document end
        if start + CHUNK_SIZE >= len(words):
            break

    return chunks


def load_policy_chunks():
    chunks = []

    for path in sorted(POLICY_DIR.glob("*.txt")):
        chunks.extend(chunk_policy_file(path))

    return chunks


# Part D, Task 2: Build TF-IDF retrieval
def build_retriever(chunks):
    vectorizer = TfidfVectorizer()
    chunk_texts = [chunk["text"] for chunk in chunks]
    matrix = vectorizer.fit_transform(chunk_texts)

    def retrieve(query, k=3):
        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(query_vector, matrix).flatten()

        ranked_indices = scores.argsort()[::-1][:k]

        return [
            {
                "source": chunks[i]["source"],
                "score": float(scores[i]),
                "text": chunks[i]["text"],
            }
            for i in ranked_indices
        ]

    return retrieve


# Part D, Task 3: Grounded policy assistant
def answer(question):
    chunks = load_policy_chunks()
    retrieve = build_retriever(chunks)
    results = retrieve(question, k=3)

    context = "\n\n".join(
        f"[{item['source']}]\n{item['text']}"
        for item in results
    )

    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT: Answer only using the supplied policy "
                "context. Do not invent policy details. If the context "
                "does not contain the answer, say exactly: "
                "I could not find that information in the policy documents."
            ),
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}",
        },
    ]

    response = mock_llm.chat(messages)

    return {
        "answer": response.choices[0].message.content or "",
        "sources": [item["source"] for item in results],
    }


# Part D, Task 3: Evaluate the 14 supplied questions
def evaluate_zero_overlap():
    global OVERLAP
    original_overlap = OVERLAP

    try:
        OVERLAP = 0
        print("\n=== ZERO-OVERLAP EVALUATION ===")
        evaluate_policy_assistant()
    finally:
        OVERLAP = original_overlap
def evaluate_policy_assistant():
    with EVAL_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    total_answerable = 0
    correct_doc = 0
    correct_fact = 0
    total_unanswerable = 0
    correct_fallback = 0

    for row in rows:
        question = row["question"].strip()
        expected_doc = row["expected_doc"].strip()
        expected_fact = row["expected_fact"].strip()

        result = answer(question)
        sources = result["sources"]
        response_text = result["answer"]

        # Answerable questions have both expected_doc and expected_fact
        if expected_doc and expected_fact:
            total_answerable += 1

            if expected_doc in sources:
                correct_doc += 1

            if expected_fact in response_text:
                correct_fact += 1
            else:
                print("\nMISSED FACT")
                print("Question:", question)
                print("Expected:", expected_fact)
                print("Answer:", response_text)

        # Unanswerable questions have empty expected fields
        else:
            total_unanswerable += 1

            if response_text.startswith("I could not find"):
                correct_fallback += 1

    print("\nPolicy Assistant Evaluation")
    print(f"Total questions: {len(rows)}")
    print(
        "Answerable — expected document retrieved: "
        f"{correct_doc}/{total_answerable}"
    )
    print(
        "Answerable — expected fact found in answer: "
        f"{correct_fact}/{total_answerable}"
    )
    print(
        "Unanswerable — correct fallback: "
        f"{correct_fallback}/{total_unanswerable}"
    )


if __name__ == "__main__":
    chunks = load_policy_chunks()
    print("Total chunks:", len(chunks))

    evaluate_policy_assistant()
