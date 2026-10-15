# PaperQA｜Hybrid RAG for Research Paper Question Answering

本專案建立一套以 **Retrieval-Augmented Generation（RAG）** 為核心的論文問答系統，結合 **FAISS Dense Retrieval、BM25 Sparse Retrieval、Reciprocal Rank Fusion（RRF）與 Cross-Encoder Reranking**，從單篇 NLP 論文中找出與問題最相關的 evidence，再由 **Llama-3.2-3B-Instruct** 根據檢索內容生成回答。

本專案由生成式人工智慧課程的 Document QA 實作延伸整理，重點探索不同 Retrieval、Fusion 與 Reranking 策略如何影響 evidence-grounded question answering。

## Highlights

- **Hybrid Retrieval**：結合 Dense Retrieval 與 BM25，同時考慮語意相關性與關鍵字匹配。
- **Reciprocal Rank Fusion**：透過 RRF 融合 Dense 與 Sparse Retrieval 的排名，避免直接比較不同尺度的 retrieval scores。
- **Two-stage Retrieval**：先取得候選段落，再利用 Cross-Encoder 進行更精細的 relevance scoring。
- **Evidence-grounded Generation**：最終回答根據選出的 evidence 生成，使回答內容具有可追溯依據。
- **Pipeline Iteration**：從 baseline dense retrieval，逐步加入 hybrid search、reranking 與 RRF。

**Tech Stack:** Python · LangChain · Sentence Transformers · FAISS · BM25 · Cross-Encoder · vLLM · Llama-3.2-3B-Instruct · ROUGE-L

---

## System Architecture

目前主要使用 `rag_pipeline_rrf.py`：

```text
Single Paper full_text
        │
        ▼
Recursive Text Chunking
300 characters / 150 overlap
        │
        ├───────────────┐
        ▼               ▼
 BGE Embeddings       BM25
        │               │
      FAISS             │
        │               │
        └───────┬───────┘
                ▼
      Reciprocal Rank Fusion
                │
                ▼
       Up to 50 Candidates
                │
                ▼
       BGE Cross-Encoder
            Reranking
                │
                ▼
          Top-1 Evidence
                │
                ▼
     Llama-3.2-3B-Instruct
                │
                ▼
      Answer + Evidence
```

每一題只會在該題所對應的單篇論文 `full_text` 中進行搜尋，不會跨論文建立索引。

---

## Retrieval Pipeline

### 1. Document Chunking

RRF 主版本將每篇論文切分為：

- Chunk size：`300` characters
- Overlap：`150` characters

---

### 2. Dense Retrieval

使用：

```text
BAAI/bge-large-en-v1.5
```

將文本轉換為 normalized embeddings，並透過 FAISS `IndexFlatIP` 計算 query 與 document chunks 之間的相似度。

---

### 3. BM25 Retrieval

同時建立 BM25 sparse retrieval。

文本使用正則式：

```text
\w+
```

進行 tokenization 並轉為小寫，再根據關鍵字匹配程度取得 BM25 ranking。

---

### 4. Reciprocal Rank Fusion

Dense Retrieval 與 BM25 不直接比較原始分數，而是使用 RRF 融合兩者排名：

```text
score(chunk)
=
1 / (60 + vector_rank)
+
1 / (60 + bm25_rank)
```

依融合後的排名，保留最多 50 個候選 chunks。

---

### 5. Cross-Encoder Reranking

候選段落使用：

```text
BAAI/bge-reranker-v2-m3
```

評估：

```text
(question, chunk)
```

之間的相關性，並重新排序候選段落，最後選出 **Top-1 evidence**。

---

### 6. Answer Generation

生成模型使用：

```text
meta-llama/Llama-3.2-3B-Instruct
```

LLM 根據最終選出的 evidence 生成回答，最終輸出：

```text
title
answer
evidence
```

---

## Pipeline Evolution

本專案保留不同開發階段，用來呈現 RAG Pipeline 的逐步改進。

| Version | Retrieval | Reranker | Chunk Size / Overlap | Evidence |
| --- | --- | --- | --- | --- |
| `rag_pipeline_baseline.py` | FAISS Dense Retrieval | — | 512 / 64 | Top-5 |
| `rag_pipeline_hybrid.py` | Dense + BM25 weighted fusion | BGE reranker | 800 / 200 | ≤ 5 |
| `rag_pipeline_rerank.py` | Dense + BM25 | BGE reranker | 400 / 100 | ≤ 3 |
| `rag_pipeline_rrf.py` | Dense + BM25 + RRF | BGE reranker v2-m3 | 300 / 150 | Top-1 |

所有版本皆使用：

```text
meta-llama/Llama-3.2-3B-Instruct
```

作為最終生成模型。

目前主要版本為：

```text
rag_pipeline_rrf.py
```

其流程為：

```text
Dense Retrieval
      +
BM25 Retrieval
      ↓
Reciprocal Rank Fusion
      ↓
Cross-Encoder Reranking
      ↓
Top-1 Evidence
      ↓
Llama Generation
```

---

## Why RRF?

Dense Retrieval 與 BM25 的分數尺度不同，因此直接將兩種分數加權相加，可能使其中一種 retrieval method 主導結果。

RRF 不直接使用原始分數，而是根據排名位置進行融合：

```text
Dense Retrieval
        +
BM25 Retrieval
        │
        ▼
Ranking Fusion
        │
        ▼
Cross-Encoder Reranking
```

這樣可以先擴大候選 evidence 的涵蓋範圍，再利用 reranker 重新排序。

---

## Dataset

原始實驗包含：

- 100 筆具有 reference answer 與 evidence 的 public QA samples
- 100 筆不提供 reference answer 的 private QA samples

每筆資料主要包含：

| Field | Description |
| --- | --- |
| `title` | Paper title |
| `full_text` | Full paper text |
| `question` | Question |
| `answer` | Reference answer, public data only |
| `evidence` | Reference evidence, public data only |

原始資料集不包含在此 GitHub repository 中。

---

## Installation

建議使用 Python 3.10 以上版本。

建立虛擬環境：

```bash
python -m venv .venv
```

Linux / macOS：

```bash
source .venv/bin/activate
```

Windows：

```bash
.venv\Scripts\activate
```

安裝 dependencies：

```bash
python -m pip install -r requirements.txt
```

主要套件：

```text
torch
numpy
faiss-cpu
sentence-transformers
rank-bm25
langchain-openai
langchain-core
langchain-community
langchain-text-splitters
rouge-score
rich
```

---

## LLM Service

Pipeline 使用 LangChain `ChatOpenAI` 連接本機 OpenAI-compatible vLLM server。

| Setting | Value |
| --- | --- |
| API Base URL | `http://localhost:8091/v1` |
| Model | `meta-llama/Llama-3.2-3B-Instruct` |
| Temperature | `0.1` |
| Max Output Tokens | `512` |

執行 pipeline 前需先啟動對應的 vLLM service。

若使用不同 host 或 port，可修改：

```python
VLLM_HOST
VLLM_PORT
```

---

## Usage

### Public Dataset

```bash
python rag_pipeline_rrf.py --dataset public
```

### Private Dataset

```bash
python rag_pipeline_rrf.py --dataset private
```

### Both

```bash
python rag_pipeline_rrf.py --dataset both
```

完整流程：

```text
Load Paper
→ Chunking
→ Dense Retrieval
→ BM25 Retrieval
→ RRF
→ Cross-Encoder Reranking
→ Evidence Selection
→ LLM Generation
→ Save Result
```

---

## Evaluation

### Evidence Evaluation

使用 ROUGE-L 評估 retrieved evidence 與 reference evidence 的文字重疊程度。

```bash
python rag_pipeline_rrf.py --eval-only
```

### Answer Evaluation

`score_public.py` 可使用 Llama-3.2-3B-Instruct 作為開發階段的 LLM judge。

```bash
python score_public.py public_results.json \
  --dataset public_dataset.json \
  --host localhost \
  --port 8091 \
  --times 5
```

每題預設評估 5 次，再透過 majority voting 決定結果。

此評估僅作為開發階段的近似指標，不視為官方 benchmark score。

---

## Output Format

輸出為 JSON：

```json
[
  {
    "title": "Example paper title",
    "answer": "A concise answer grounded in the retrieved evidence.",
    "evidence": [
      "A relevant excerpt retrieved from the paper."
    ]
  }
]
```

GitHub repository 中僅保留少量範例：

```text
examples/sample_output.json
```

---


## Project Structure

```text
PaperQA-Hybrid-RAG/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── rag_pipeline_baseline.py
├── rag_pipeline_hybrid.py
├── rag_pipeline_rerank.py
├── rag_pipeline_rrf.py
│
├── score_public.py
│
└── examples/
    └── sample_output.json
```

---

## Key Takeaways

這個專案的重點不只是呼叫 LLM，而是設計 **LLM 回答之前的資訊檢索與 evidence selection 流程**。

整體架構可以概括為：

```text
Retrieve
   ↓
Fuse
   ↓
Rerank
   ↓
Ground
   ↓
Generate
```

本專案透過不同 retrieval pipeline 的迭代，探索 **Chunking、Dense Retrieval、Sparse Retrieval、Ranking Fusion、Reranking 與 Evidence Selection** 在 RAG 系統中的角色。