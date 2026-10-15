"""
HW2: Document QA based on RAG
Baseline pipeline using:
- Embedding: BAAI/bge-small-en-v1.5
- Vector store: FAISS (per-paper)
- LLM: meta-llama/Llama-3.2-3B-Instruct via vLLM
- Chunking: RecursiveCharacterTextSplitter
"""

import json
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.progress import track

console = Console()

# ── Config ────────────────────────────────────────────────────────────────────
VLLM_HOST = "localhost"
VLLM_PORT = 8091
VLLM_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 5  # number of chunks to retrieve per question

PUBLIC_DATASET = "public_dataset.json"
PRIVATE_DATASET = "private_dataset.json"

# ── Initialize models ────────────────────────────────────────────────────────
console.print("[bold blue]Loading embedding model...[/bold blue]")
embed_model = SentenceTransformer(EMBED_MODEL, device="cuda")

console.print("[bold blue]Connecting to vLLM server...[/bold blue]")
llm = ChatOpenAI(
    model=VLLM_MODEL,
    base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
    api_key="not-needed",
    temperature=0.1,
    max_tokens=256,
)

# ── Text splitter ────────────────────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
)

# ── Prompt ────────────────────────────────────────────────────────────────────
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful research assistant. Answer the question based ONLY on the provided context from the paper. Be concise and accurate. If the context does not contain enough information, say so."),
    ("human", """Paper title: {title}

Context from paper:
{context}

Question: {question}

Answer:"""),
])

qa_chain = QA_PROMPT | llm | StrOutputParser()


def chunk_and_index(full_text: str) -> tuple[list[str], faiss.IndexFlatIP]:
    """Split a paper's full text into chunks and build a FAISS index."""
    chunks = text_splitter.split_text(full_text)
    if not chunks:
        return [], None

    # BGE models recommend prefixing for retrieval
    embeddings = embed_model.encode(
        chunks,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized = cosine
    index.add(embeddings)

    return chunks, index


def retrieve(question: str, chunks: list[str], index: faiss.IndexFlatIP, top_k: int = TOP_K) -> list[str]:
    """Retrieve top-k chunks most relevant to the question."""
    if not chunks or index is None:
        return []

    # BGE recommends query prefix for retrieval
    q_emb = embed_model.encode(
        [f"Represent this sentence for searching relevant passages: {question}"],
        normalize_embeddings=True,
    )
    q_emb = np.array(q_emb, dtype=np.float32)

    k = min(top_k, len(chunks))
    scores, indices = index.search(q_emb, k)

    return [chunks[i] for i in indices[0] if i < len(chunks)]


def answer_question(title: str, question: str, evidence: list[str]) -> str:
    """Use the LLM to answer a question given retrieved evidence."""
    context = "\n\n---\n\n".join(evidence)
    try:
        result = qa_chain.invoke({
            "title": title,
            "question": question,
            "context": context,
        })
        return result.strip()
    except Exception as e:
        console.print(f"[red]LLM error: {e}[/red]")
        return "Unable to answer."


def process_dataset(dataset_path: str, output_path: str):
    """Process a full dataset and save results."""
    with open(dataset_path) as f:
        dataset = json.load(f)

    console.print(f"[bold green]Processing {len(dataset)} papers from {dataset_path}[/bold green]")

    results = []
    for i, paper in enumerate(track(dataset, description="Processing papers...")):
        title = paper["title"]
        full_text = paper["full_text"]
        question = paper["question"]

        # Step 1: Chunk and index the paper
        chunks, index = chunk_and_index(full_text)

        # Step 2: Retrieve relevant chunks
        evidence = retrieve(question, chunks, index, top_k=TOP_K)

        # Step 3: Answer using LLM
        answer = answer_question(title, question, evidence)

        results.append({
            "title": title,
            "answer": answer,
            "evidence": evidence,
        })

        if (i + 1) % 10 == 0:
            console.print(f"  [{i+1}/{len(dataset)}] Last answer: {answer[:100]}...")

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print(f"[bold green]Saved {len(results)} results to {output_path}[/bold green]")
    return results


def evaluate_public(results_path: str):
    """Quick self-evaluation using ROUGE-L for evidence score on public dataset."""
    from rouge_score import rouge_scorer

    with open(PUBLIC_DATASET) as f:
        gt_data = {item["title"]: item for item in json.load(f)}

    with open(results_path) as f:
        results = json.load(f)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    evidence_scores = []
    for item in results:
        title = item["title"]
        gt = gt_data.get(title)
        if gt is None:
            continue

        retrieved = item.get("evidence", [])
        golden = gt.get("evidence", [])

        if not golden:
            evidence_scores.append(1.0)
            continue
        if not retrieved:
            evidence_scores.append(0.0)
            continue

        score = sum(
            scorer.score_multi(targets=golden, prediction=c)["rougeL"].fmeasure
            for c in retrieved
        ) / len(retrieved)
        evidence_scores.append(score)

    avg_evidence = sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0
    console.print(f"\n[bold yellow]Evidence Score (ROUGE-L): {avg_evidence:.4f}[/bold yellow]")
    console.print(f"  Weak baseline: 0.2124 | Strong baseline: 0.26185")
    return avg_evidence


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["public", "private", "both"], default="public")
    parser.add_argument("--eval", action="store_true", help="Run evidence evaluation on public results")
    args = parser.parse_args()

    if args.dataset in ("public", "both"):
        process_dataset(PUBLIC_DATASET, "public_results.json")
        evaluate_public("public_results.json")

    if args.dataset in ("private", "both"):
        process_dataset(PRIVATE_DATASET, "private_results.json")

    if args.eval:
        evaluate_public("public_results.json")
