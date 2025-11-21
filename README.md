
```markdown
# Crowdfunding Insights RAG System

## Overview

This project implements a **Retrieval-Augmented Generation (RAG) system** to analyze Indian crowdfunding campaigns (scraped from Ketto) and provide **insights from both numeric and textual data**.  

The system is designed for **speed, accuracy, and hybrid query handling**, combining **FAISS embeddings**, **open-source embeddings**, and **LLMs**. It supports both **semantic (text)** and **numeric/statistical queries**.

Key features:  
- Scraped, cleaned, and enriched crowdfunding campaign dataset.  
- Summaries: numeric, text, and overall metrics.  
- Hybrid RAG system with **local open-source models** and **Groq cloud LLM**.  
- Efficient FAISS-based retrieval for large datasets.  

---

## 📂 Project Structure

```
<img width="1896" height="797" alt="image" src="https://github.com/user-attachments/assets/d89df937-5346-4dde-94d4-4aefda76d5ce" />

Impactguru/
├─ docs_json/                    # JSON documents for each campaign
├─ faiss_index/                  # FAISS vectorstore for embeddings
├─ models/                       # Local open-source LLMs
│   ├─ llama-pro-8b-instruct.Q4_K_S.gguf
│   └─ Mistral-7B-Instruct-v0.3.Q4_K_S.gguf
├─ src/                          # Source scripts
│   ├─ infer.py                  # Interactive CLI for hybrid queries
│   ├─ local_rag.py              # Local RAG implementation using downloaded models
│   ├─ rag.py                    # Main RAG system script
│   ├─ raghybrid.py              # Hybrid RAG functions + helpers
│   ├─ scrapper.py               # Web scraping script for crowdfunding campaigns
│   └─ summariser.py             # Generates numeric/text/overall summaries
├─ ketto_campaigns.csv            # Raw scraped campaign dataset
├─ ketto_campaigns_enriched.csv  # Processed & enriched dataset
├─ numeric_summary.csv            # Numeric statistics
├─ text_summary.txt               # Text story summaries
├─ overall_summary.txt            # Overall metrics
├─ streamlit.py                   # Streamlit interface for hybrid queries
├─ requirements.txt               # Python dependencies
└─ README.md


````

---

## 1️⃣ Data Pipeline

**Source:**  
- Crowdfunding campaigns scraped from [Ketto](https://www.ketto.org) using `scrapper.py`.  

**Processing Steps:**  
1. **Cleaning:** Handle missing numeric/text fields, standardize empty values.  
2. **Enrichment:** Add metadata like `Campaign URL`, `Campaigner`, `Age Group`, `Raised`, `Goal`, `Avg Donation`, `Days Left`, `Urgency`, `Funding Gap`, `Sentiment`, `Story Summary`.  
3. **JSON Generation:** Each campaign converted into a separate JSON in `docs_json/`.  
4. **Summaries:** `summariser.py` generates:  
   - Numeric summary (`numeric_summary.csv`)  
   - Text summary (`text_summary.txt`)  
   - Overall summary (`overall_summary.txt`)  

---

## 2️⃣ RAG System Design

**Components:**  

| Component          | Choice                                  | Reasoning |
| -----------------  | -------------------------------------- | --------- |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` | Lightweight, fast, open-source, effective for semantic similarity. |
| **Vectorstore**     | FAISS                                   | Fast top-K similarity search, in-memory, scalable. |
| **Local LLM**       | `Mistral-7B-Instruct` / `LLaMA-pro-8B`  | Open-source, instruction-tuned, supports offline queries. |
| **Cloud LLM**       | `ChatGroq`                              | Fast inference for hybrid queries, scalable, and cloud-hosted. |
| **Chunking**        | `RecursiveCharacterTextSplitter`       | Efficiently splits long campaign stories while preserving context. |

**Query Handling:**  
1. **Numeric Queries:**  
   - Precomputed aggregates (`sum`, `mean`, `max`, `min`, `count`) for numeric fields.  
   - Derived metrics: `% of Goal Raised`, `Avg Donation per Supporter`.  
   - Example: "Who has raised the most?", "Average supporters?"  

2. **Semantic Queries:**  
   - FAISS retrieves top-K relevant chunks of campaign stories.  
   - LLM (Groq or local) synthesizes answers.  
   - Example: "Which campaigns are urgent or medical-related?"  

**Performance Decision:**  
- Local models were initially used but **slowed down inference** for large datasets.  
- Switched to **Groq cloud LLM** for **fast, high-quality responses**, while keeping local models for offline testing.

---

## 3️⃣ How to Run

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
````

### Step 2: Scrape campaigns (if needed)

```bash
python src/scrapper.py
```

* Outputs raw campaign CSV (`ketto_campaigns.csv`).

### Step 3: Generate summaries

```bash
python src/summariser.py
```

* Produces numeric, text, and overall summaries.

### Step 4: Build RAG system

```bash
python src/rag.py
```

* Loads campaigns.
* Creates enriched documents & chunks.
* Builds FAISS vectorstore.
* Computes numeric aggregates.
* RAG system ready for queries.

### Step 5: Query interactively

```bash
python src/infer.py
```

* Ask numeric/statistical or semantic questions.
* Use metadata filters (`Campaigner`, `Age Group`, `Urgency`) for refined answers.

### Optional: Web Interface

```bash
streamlit run src/streamlit.py
```

* Web-based interface for easy hybrid query access.

---

## 4️⃣ Features

* **Hybrid Numeric + Semantic Queries**
* **Metadata Filtering:** Filter campaigns by urgency, age group, campaigner, etc.
* **Derived Metrics:** `% of Goal Raised`, `Funding Gap`, `Avg Donation per Supporter`.
* **Interactive CLI & Streamlit Interface**
* **Fast Performance:** Cloud LLM (Groq) for responsive answers.
* **Extensible:** Add new campaigns or models easily.

---

## 5️⃣ References

* [LangChain](https://www.langchain.com/)
* [FAISS](https://github.com/facebookresearch/faiss)
* [Sentence Transformers](https://www.sbert.net/)
* [Mistral 7B](https://mistral.ai/models)
* [LLaMA-pro 8B](https://huggingface.co/)
* [Ketto](https://www.ketto.org)


# Key Summary
## 📌 Model Comparison (Local + Groq Hosted)

| Model Name | Source | Type | Speed | Ideal For | Notes |
|------------|--------|-------|--------|------------|--------|
| **Mistral-7B-Instruct v0.3** | Local | Open-Source | Medium | Local RAG, CPU/GPU | Apache-2.0 license, strong quality. |
| **Phi-3 Mini 4K Instruct (3.8B)** | Local | Open-Source | Fast | Lightweight RAG, CPU-friendly | MIT license, very low memory usage. |
| **meta-llama/llama-4-maverick-17b-128e-instruct** | Groq | Open-Weight | Ultra Fast | High-quality RAG | Excellent reasoning, <1s latency on Groq. |
| **qwen/qwen3-32b-instruct** | Groq | Open-Weight | Ultra Fast | Deep insights, multilingual RAG | Very strong performance across tasks. |
| **llama-3.1-8b-instant** | Groq | Open-Weight | Ultra Fast | General RAG | Good balance of speed + quality. |
| **llama-3.3-70b-versatile** | Groq | Open-Weight | Fast | Complex reasoning | Highest accuracy among LLaMA family. |
| **moonshotai/kimi-k2-instruct** | Groq | Open-Weight | Fast | General chat + RAG | Good multilingual support. |
| **openai/gpt-oss-20b** | Groq | Open-Weight | Fast | RAG + reasoning | Open-weight variant, strong accuracy. |
| **openai/gpt-oss-120b** | Groq | Open-Weight | Medium | Long-context reasoning | High capability but slower. |
                      


Local open-source models (Mistral-7B and Phi-3 Mini) provide full control and privacy,  
but were slow on CPU (≈3–5 minutes per query).  

To achieve real-time RAG performance, I switched to Groq-hosted open-weight models  
(LLaMA-4 Maverick 17B and Qwen3-32B), which provide sub-second latency while  
remaining fully within open-source/open-weight constraints.


---


```
