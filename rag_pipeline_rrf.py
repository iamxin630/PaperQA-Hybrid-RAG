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
from rouge_score import rouge_scorer
from rich.console import Console
from rich.progress import track

console = Console()

# ── Config ────────────────────────────────────────────────────────────────────
VLLM_HOST = "localhost"
VLLM_PORT = 8091
VLLM_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

CHUNK_SIZE = 300          # 與 golden evidence 大小匹配（median ~216 chars）
CHUNK_OVERLAP = 150       # 50% 重疊
MIN_CHUNK_LEN = 30        # 過濾無效的文本塊
INITIAL_TOP_K = 50        # 更多候選區塊提升召回率
FINAL_TOP_K = 1           # 固定取 TOP-1

RRF_K = 60                # RRF演算法中的常數

PUBLIC_DATASET = "public_dataset.json"
PRIVATE_DATASET = "private_dataset.json"

# ── Initialize models ────────────────────────────────────────────────────────
console.print("[bold blue]Loading embedding model...[/bold blue]")
embed_model = SentenceTransformer(EMBED_MODEL, device="cuda") # 載入模型，負責將一段文字轉換成一串數字向量

console.print("[bold blue]Loading reranker model...[/bold blue]")
reranker = CrossEncoder(RERANK_MODEL, device="cpu") # 載入模型，能同時輸入問題與檢索到的區塊，並計算兩者的關聯得分

console.print("[bold blue]Connecting to vLLM server...[/bold blue]")
# 建立一個與 vLLM 伺服器溝通的橋樑
llm = ChatOpenAI(
    model=VLLM_MODEL,
    base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
    api_key="not-needed",
    temperature=0.1, # 讓模型的回答變得穩定且保守，避免 Hallucination
    max_tokens=512,
)

# ── Text splitter ────────────────────────────────────────────────────────────
# 根據 separators 列表中的順序，遞迴尋找適合切斷的位置
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n\n", "\n\n", "\n", ". ", "; ", ", ", " ", ""],
)

# ── Prompts ───────────────────────────────────────────────────────────────────
# HyDE: 生成假設答案，用來做 embedding 搜尋（比直接用問題更貼近 evidence 段落）
HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Write a single concise sentence that directly answers the question, as if it came from the paper itself. Output only the answer sentence, nothing else."),
    ("human", "Paper title: {title}\nQuestion: {question}\n\nAnswer sentence:"),
])
hyde_chain = HYDE_PROMPT | llm | StrOutputParser()

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise research paper analyst. Answer ONLY from the provided excerpts.\n"
     "Rules:\n"
     "1. Give a complete, specific answer in 1-3 sentences.\n"
     "2. NEVER say 'not mentioned', 'not explicitly stated', 'unclear', or 'I don't know'. Always extract the best answer from the text.\n"
     "3. If the answer is a list, include ALL items mentioned.\n"
     "4. Use the EXACT terms, names, numbers, and metrics from the excerpts.\n"
     "5. Be concise but complete — do not omit important details."),
    ("human", "Paper: {title}\n\nExcerpts:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
])

qa_chain = QA_PROMPT | llm | StrOutputParser()


def generate_hyde_query(title: str, question: str) -> str:
    """Generate a hypothetical answer sentence for better vector retrieval."""
    try:
        result = hyde_chain.invoke({"title": title, "question": question})
        return result.strip()
    except Exception:
        return question  # fallback to original question

# 將原始句子轉化為單字 (BM25Okapi)
def tokenize_simple(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def chunk_and_index(full_text: str) -> tuple[list[str], faiss.IndexFlatIP, BM25Okapi | None]:
    # Text Chunking
    chunks = text_splitter.split_text(full_text)
    chunks = [c for c in chunks if len(c.strip()) >= MIN_CHUNK_LEN]
    if not chunks:
        return [], None, None

    # Generating Embeddings
    embeddings = embed_model.encode(
        chunks,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    # Vector Indexing
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) # 建立內積平面索引
    index.add(embeddings)

    # BM25 Indexing
    tokenized_chunks = [tokenize_simple(c) for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    return chunks, index, bm25


def hybrid_retrieve_with_indices(
    question: str,
    chunks: list[str],
    index: faiss.IndexFlatIP,
    bm25: BM25Okapi,
    top_k: int = INITIAL_TOP_K,
) -> list[tuple[int, str]]:
    """Retrieve top-k chunks using RRF fusion. Returns (index, text) pairs."""
    # 如果沒有資料或索引未建立，直接回傳空清單
    if not chunks or index is None:
        return []

    n_chunks = len(chunks) # 總共有多少個區塊
    k = min(top_k, n_chunks) # 確保輸出的 k 不會超過總區塊數

    # Vector search → rankings
    q_emb = embed_model.encode(
        [f"Represent this sentence for searching relevant passages: {question}"],
        normalize_embeddings=True,
    )
    # 轉為 float32 以符合 FAISS 要求
    q_emb = np.array(q_emb, dtype=np.float32)

    # 在向量索引中搜尋所有區塊 (n_chunks)，目的是得到完整排名
    _, vec_indices = index.search(q_emb, n_chunks)
    vec_indices = vec_indices[0]
    
    # 建立一個「索引 -> 排名」的對照表
    vec_ranks = {}
    for rank, idx in enumerate(vec_indices):
        vec_ranks[int(idx)] = rank

    # BM25 search → rankings
    # 將問題斷詞，轉為小寫單字清單
    q_tokens = tokenize_simple(question)

    # 計算問題與每個區塊的 BM25 分數
    bm25_scores = bm25.get_scores(q_tokens)

    # 根據分數從高到低進行排序 (argsort 取負號代表降序)
    bm25_ranked = np.argsort(-bm25_scores)
    
    # 同樣建立一個「索引 -> 排名」的對照表 (bm25_ranks)
    bm25_ranks = {}
    for rank, idx in enumerate(bm25_ranked):
        bm25_ranks[int(idx)] = rank

    # Reciprocal Rank Fusion
    combined = []
    # 遍歷每一個區塊，計算它的「融合分數」
    for i in range(n_chunks):
        vr = vec_ranks.get(i, n_chunks)
        br = bm25_ranks.get(i, n_chunks)

        # RRF 公式：得分 = 1/(k + 向量排名) + 1/(k + 關鍵字排名)
        rrf = 1.0 / (RRF_K + vr) + 1.0 / (RRF_K + br)
        combined.append((i, rrf))

    # 根據 RRF 分數由高到低排序
    combined.sort(key=lambda x: x[1], reverse=True)

    # 回傳分數最高的前 k 個結果，格式為 (原始索引, 區塊文字)
    return [(idx, chunks[idx]) for idx, _ in combined[:k]]


def rerank_with_indices(
    question: str,
    candidates: list[tuple[int, str]],
) -> list[tuple[int, str, float]]:
    """Rerank candidates. Returns sorted (index, text, score) triples."""
    # 如果傳進來的候選清單是空的，就直接回傳空列表
    if not candidates:
        return []

    # 建立一個 (問題, 區塊文本) 的對應列表
    pairs = [(question, text) for _, text in candidates]

    # 將所有資料對丟進重排模型，計算每一對的相關性分數
    scores = reranker.predict(pairs)

    # 將原始的 (索引, 文本) 與剛算出來的「分數」重新黏在一起
    ranked = sorted(
        [(idx, text, score) for (idx, text), score in zip(candidates, scores)],
        key=lambda x: x[2], # 根據第三個元素 (score) 來排序
        reverse=True, # 由高分到低分排列 (降序)
    )
    return ranked # 回傳排序後的三元組清單


def answer_question(title: str, question: str, context_chunks: list[str]) -> str:
    # Context Formatting
    context = "\n\n---\n\n".join(f"[Excerpt {i+1}]\n{e}" for i, e in enumerate(context_chunks))
    # Invoking QA Chain
    try:
        result = qa_chain.invoke({
            "title": title,
            "question": question,
            "context": context,
        })
        return result.strip()
    # 錯誤處理
    except Exception as e:
        console.print(f"[red]LLM error: {e}[/red]")
        return "Unable to answer."


def process_dataset(dataset_path: str, output_path: str):
    # 打開指定的資料集 JSON 檔案
    with open(dataset_path) as f:
        dataset = json.load(f)

    console.print(f"[bold green]Processing {len(dataset)} papers from {dataset_path}[/bold green]")

    results = []
    for i, paper in enumerate(track(dataset, description="Processing papers...")):
        title = paper["title"]
        full_text = paper["full_text"]
        question = paper["question"]

        # Step 1: Chunk and index
        chunks, index, bm25 = chunk_and_index(full_text)

        # Step 2: Hybrid retrieve with RRF
        candidates = hybrid_retrieve_with_indices(question, chunks, index, bm25)

        # Step 3: Rerank
        ranked = rerank_with_indices(question, candidates)

        # Step 4: Get evidence (fixed TOP-1)
        evidence = [text for _, text, _ in ranked[:FINAL_TOP_K]]

        # Step 5: Answer using LLM
        answer = answer_question(title, question, evidence)

        results.append({
            "title": title,
            "answer": answer,
            "evidence": evidence,
        })

        if (i + 1) % 10 == 0:
            console.print(f"  [{i+1}/{len(dataset)}] answer: {answer[:80]}...")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print(f"[bold green]Saved {len(results)} results to {output_path}[/bold green]")
    return results


def evaluate_public(results_path: str):
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
