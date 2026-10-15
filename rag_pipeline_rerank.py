"""
HW2: Document QA based on RAG — v3 (optimized evidence selection)

Key insight: evidence score = avg ROUGE-L over retrieved chunks.
Returning fewer, higher quality chunks maximizes the score.

- Embedding: BAAI/bge-large-en-v1.5
- Reranker: BAAI/bge-reranker-base (CPU)
- Hybrid search: FAISS + BM25
- Return only top 1-3 evidence chunks (reranker-selected)
- LLM: meta-llama/Llama-3.2-3B-Instruct via vLLM
"""

import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.progress import track

console = Console()

# ── Config ────────────────────────────────────────────────────────────────────
VLLM_HOST = "localhost"
VLLM_PORT = 8091
VLLM_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
INITIAL_TOP_K = 15        # retrieve more candidates for reranking
FINAL_TOP_K = 3           # max evidence chunks after reranking (lower = higher avg quality)
RERANK_DROP_RATIO = 0.6   # drop chunks scoring below this fraction of top score

VECTOR_WEIGHT = 0.5
BM25_WEIGHT = 0.5

PUBLIC_DATASET = "public_dataset.json"
PRIVATE_DATASET = "private_dataset.json"

# ── Initialize models ────────────────────────────────────────────────────────
console.print("[bold blue]Loading embedding model...[/bold blue]")
embed_model = SentenceTransformer(EMBED_MODEL, device="cuda")

console.print("[bold blue]Loading reranker model...[/bold blue]")
reranker = CrossEncoder(RERANK_MODEL, device="cpu")

console.print("[bold blue]Connecting to vLLM server...[/bold blue]")
llm = ChatOpenAI(
    model=VLLM_MODEL,
    base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
    api_key="not-needed",
    temperature=0.1,
    max_tokens=512,
)

# ── Text splitter ────────────────────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
)

# ── Prompt ────────────────────────────────────────────────────────────────────
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise research assistant. Answer the question based ONLY on the provided context excerpts from the paper. "
     "Give a direct, concise answer. Do not speculate beyond what the context states."),
    ("human", """Paper: {title}

Context:
{context}

Question: {question}

Answer concisely based on the context:"""),
])

qa_chain = QA_PROMPT | llm | StrOutputParser()


def tokenize_simple(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def chunk_and_index(full_text: str) -> tuple[list[str], faiss.IndexFlatIP, BM25Okapi | None]:
    chunks = text_splitter.split_text(full_text)
    if not chunks:
        return [], None, None

    embeddings = embed_model.encode(
        chunks,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    tokenized_chunks = [tokenize_simple(c) for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    return chunks, index, bm25


def hybrid_retrieve(
    question: str,
    chunks: list[str],
    index: faiss.IndexFlatIP,
    bm25: BM25Okapi,
    top_k: int = INITIAL_TOP_K,
) -> list[str]:
    if not chunks or index is None:
        return []

    n_chunks = len(chunks)
    k = min(top_k, n_chunks)

    # Vector search
    q_emb = embed_model.encode(
        [f"Represent this sentence for searching relevant passages: {question}"],
        normalize_embeddings=True,
    )
    q_emb = np.array(q_emb, dtype=np.float32)
    vec_scores, vec_indices = index.search(q_emb, n_chunks)
    vec_scores = vec_scores[0]
    vec_indices = vec_indices[0]

    vec_score_map = {}
    vs_min, vs_max = vec_scores.min(), vec_scores.max()
    if vs_max > vs_min:
        for idx, score in zip(vec_indices, vec_scores):
            vec_score_map[int(idx)] = (score - vs_min) / (vs_max - vs_min)
    else:
        for idx in vec_indices:
            vec_score_map[int(idx)] = 1.0

    # BM25 search
    q_tokens = tokenize_simple(question)
    bm25_scores = bm25.get_scores(q_tokens)
    b_min, b_max = bm25_scores.min(), bm25_scores.max()
    if b_max > b_min:
        norm_bm25 = (bm25_scores - b_min) / (b_max - b_min)
    else:
        norm_bm25 = np.ones_like(bm25_scores)

    # Combine
    combined = []
    for i in range(n_chunks):
        vs = vec_score_map.get(i, 0.0)
        bs = norm_bm25[i]
        combined.append((i, VECTOR_WEIGHT * vs + BM25_WEIGHT * bs))

    combined.sort(key=lambda x: x[1], reverse=True)
    return [chunks[idx] for idx, _ in combined[:k]]


def rerank_and_select(
    question: str,
    candidates: list[str],
    max_k: int = FINAL_TOP_K,
) -> list[str]:
    if not candidates:
        return []

    pairs = [(question, text) for text in candidates]
    scores = reranker.predict(pairs)

    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top_score = scored[0][1]

    # Keep chunks with score within drop_ratio of the top score
    # Always keep at least 1
    selected = [scored[0][0]]
    for text, score in scored[1:max_k]:
        # Keep if score is not too far below the best
        if score >= top_score * RERANK_DROP_RATIO:
            selected.append(text)

    return selected


def answer_question(title: str, question: str, evidence: list[str]) -> str:
    context = "\n\n---\n\n".join(f"[Excerpt {i+1}]\n{e}" for i, e in enumerate(evidence))
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
    with open(dataset_path) as f:
        dataset = json.load(f)

    console.print(f"[bold green]Processing {len(dataset)} papers from {dataset_path}[/bold green]")

    results = []
    evid_counts = []
    for i, paper in enumerate(track(dataset, description="Processing papers...")):
        title = paper["title"]
        full_text = paper["full_text"]
        question = paper["question"]

        chunks, index, bm25 = chunk_and_index(full_text)
        candidates = hybrid_retrieve(question, chunks, index, bm25)
        evidence = rerank_and_select(question, candidates)
        answer = answer_question(title, question, evidence)

        results.append({
            "title": title,
            "answer": answer,
            "evidence": evidence,
        })
        evid_counts.append(len(evidence))

        if (i + 1) % 10 == 0:
            avg_k = sum(evid_counts) / len(evid_counts)
            console.print(f"  [{i+1}/{len(dataset)}] avg_k={avg_k:.1f} answer: {answer[:80]}...")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print(f"[bold green]Saved {len(results)} results to {output_path}[/bold green]")
    avg_k = sum(evid_counts) / len(evid_counts)
    console.print(f"  Average evidence count: {avg_k:.2f}")
    return results


def evaluate_public(results_path: str):
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

    below_02 = sum(1 for s in evidence_scores if s < 0.2)
    above_03 = sum(1 for s in evidence_scores if s > 0.3)
    console.print(f"  Below 0.2: {below_02}/100  |  Above 0.3: {above_03}/100")
    return avg_evidence


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["public", "private", "both"], default="public")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if args.eval_only:
        evaluate_public("public_results.json")
        sys.exit(0)

    if args.dataset in ("public", "both"):
        process_dataset(PUBLIC_DATASET, "public_results.json")
        evaluate_public("public_results.json")

    if args.dataset in ("private", "both"):
        process_dataset(PRIVATE_DATASET, "private_results.json")
