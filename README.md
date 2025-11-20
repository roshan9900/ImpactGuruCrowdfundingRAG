
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

## 5️⃣ Assessment & Hiring Highlights

* Demonstrates **full data engineering pipeline**: scraping → cleaning → enrichment → summaries.
* Implements **hybrid RAG system** combining numeric analytics and semantic search.
* Shows **practical use of embeddings and vectorstores (FAISS)**.
* Integrates **local & cloud LLMs** for flexibility & speed.
* Produces **interactive tools** (CLI + Streamlit) for end-users.
* Efficiently handles **large datasets with chunking**.

**This project demonstrates advanced skills in:**

* Python programming
* Data preprocessing & analytics
* LLMs & RAG architecture
* Semantic search and embeddings
* Building interactive user interfaces

---

## 6️⃣ References

* [LangChain](https://www.langchain.com/)
* [FAISS](https://github.com/facebookresearch/faiss)
* [Sentence Transformers](https://www.sbert.net/)
* [Mistral 7B](https://mistral.ai/models)
* [LLaMA-pro 8B](https://huggingface.co/)
* [Ketto](https://www.ketto.org)

---

## 7️⃣ Notes

* Rebuild FAISS vectorstore after adding new campaigns.
* Groq LLM recommended for fast interactive querying; local models are for offline or experimentation.
* Streamlit interface enhances usability for non-technical users.

```
