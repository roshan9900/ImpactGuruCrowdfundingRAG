
import os
import json
import pandas as pd
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Open-source LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

load_dotenv()

# =============================================================
# CONFIG
# =============================================================
DATA_FOLDER = "docs_json"
VECTORSTORE_PATH = "faiss_index"

NUMERIC_SUMMARY_FILE = "numeric_summary.csv"
TEXT_SUMMARY_FILE = "text_summary.txt"
OVERALL_SUMMARY_FILE = "overall_summary.txt"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPEN_SOURCE_LLM = "mistralai/Mistral-7B-Instruct-v0.3"   # Fully open-source

NUMERIC_COLUMNS = ["Raised", "Goal", "Supporters", "Days Left", "Avg Donation"]

# =============================================================
# OPEN-SOURCE LLM PIPELINE
# =============================================================
def load_open_source_llm():
    print("[INFO] Loading open-source LLM… (Mistral 7B)")
    tokenizer = AutoTokenizer.from_pretrained(OPEN_SOURCE_LLM)
    model = AutoModelForCausalLM.from_pretrained(
        OPEN_SOURCE_LLM,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False
    )
    return gen

open_llm = load_open_source_llm()

# =============================================================
# 1. LOAD CAMPAIGNS
# =============================================================
def load_campaigns(folder_path):
    campaigns = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict):
                        campaigns.append(data)
                    elif isinstance(data, list):
                        campaigns.extend(data)
                except Exception as e:
                    print(f"[WARN] Failed to load {filename}: {e}")
    return campaigns

# =============================================================
# 2. CREATE DOCUMENTS
# =============================================================
def create_document(campaign):
    content = campaign.get("Story Summary", "") or "No story provided."
    metadata = {
        key: campaign.get(key, "")
        for key in [
            "Campaign URL", "Campaigner", "Age Group", "Raised", "Goal",
            "Supporters", "Avg Donation", "Days Left", "Urgency",
            "Funding Gap", "Sentiment"
        ]
    }
    return Document(page_content=content, metadata=metadata)

# =============================================================
# 3. SPLIT DOCUMENTS
# =============================================================
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunked = []
    for doc in documents:
        if not doc.page_content.strip():
            continue
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            chunked.append(Document(page_content=chunk, metadata=doc.metadata.copy()))
    return chunked

# =============================================================
# 4. BUILD VECTORSTORE
# =============================================================
def build_vectorstore(documents):
    print("[INFO] Building FAISS vectorstore…")
    valid_docs = [d for d in documents if d.page_content.strip()]
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = FAISS.from_documents(valid_docs, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)

    return vectorstore, embeddings

# =============================================================
# 5. SAVE SUMMARIES
# =============================================================
def save_summaries(campaigns):

    if not campaigns:
        print("[WARN] No campaigns found!")
        return

    df = pd.DataFrame(campaigns)

    # Numeric summary
    numeric_summary = df[NUMERIC_COLUMNS].describe()
    numeric_summary.to_csv(NUMERIC_SUMMARY_FILE)

    # Text summary
    with open(TEXT_SUMMARY_FILE, "w", encoding="utf-8") as f:
        for c in campaigns:
            s = c.get("Story Summary", "")
            if s:
                f.write(s + "\n\n")

    # Overall summary
    overall = f"Total Campaigns: {len(campaigns)}\n"
    overall += f"Total Raised: {df['Raised'].sum()}\n"
    overall += f"Total Goal: {df['Goal'].sum()}\n"

    with open(OVERALL_SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(overall)

# =============================================================
# 6. NUMERIC AGGREGATES
# =============================================================
def compute_numeric_aggregates(campaigns):
    df = pd.DataFrame(campaigns)

    df["Raised % of Goal"] = df.apply(
        lambda x: (x["Raised"] / x["Goal"] * 100) if x["Goal"] else 0, axis=1
    )

    agg = {}
    numeric_cols = NUMERIC_COLUMNS + ["Raised % of Goal"]

    for col in numeric_cols:
        if col in df.columns:
            agg[col] = {
                "sum": df[col].sum(),
                "mean": df[col].mean(),
                "max": df[col].max(),
                "min": df[col].min(),
                "count": df[col].count(),
                "df": df
            }
    return agg

# =============================================================
# 7. LLM ANSWER GENERATION
# =============================================================
def generate_llm_answer(prompt_value):
    output = open_llm(prompt_value.to_string())[0]["generated_text"]
    return output

# =============================================================
# 8. RAG + NUMERIC QUERY SYSTEM
# =============================================================
def get_insight_from_rag(vectorstore, embeddings, documents, numeric_aggregates, question, k=5, metadata_filter=None):

    question_l = question.lower()

    # --- numeric handling ---
    if any(w in question_l for w in ["average", "mean", "total", "sum", "max", "min", "count", "%", "percentage"]):

        # Derived column
        if "percentage" in question_l or "% of goal" in question_l:
            df = numeric_aggregates["Raised % of Goal"]["df"]
            row = df.loc[df["Raised % of Goal"].idxmax()]
            return f"Highest % of goal raised: {row['Campaigner']} ({row['Raised % of Goal']:.2f}%)"

        for col in numeric_aggregates.keys():
            if col.lower() in question_l:
                agg = numeric_aggregates[col]
                df = agg["df"]

                if "max" in question_l or "highest" in question_l:
                    row = df.loc[df[col].idxmax()]
                    return f"Highest {col}: {row['Campaigner']} ({row[col]:.2f})"
                if "min" in question_l or "lowest" in question_l:
                    row = df.loc[df[col].idxmin()]
                    return f"Lowest {col}: {row['Campaigner']} ({row[col]:.2f})"
                if "mean" in question_l or "average" in question_l:
                    return f"Average {col}: {agg['mean']:.2f}"
                if "total" in question_l or "sum" in question_l:
                    return f"Total {col}: {agg['sum']:.2f}"
                if "count" in question_l:
                    return f"Count for {col}: {agg['count']}"

        return "Numeric question detected but column not identified."

    # --- Semantic RAG ---
    search_base = vectorstore

    # metadata filter
    if metadata_filter:
        filtered_docs = [
            d for d in documents
            if all(str(d.metadata.get(k, "")).lower() == str(v).lower()
                   for k, v in metadata_filter.items())
        ]
        if not filtered_docs:
            return "No documents match the filter."

        search_base = FAISS.from_documents(filtered_docs, embeddings)

    results = search_base.similarity_search(question, k=k)

    if not results:
        return "No relevant documents found."

    combined_text = "\n\n".join([r.page_content for r in results])

    prompt = ChatPromptTemplate.from_template("""
You are a crowdfunding insights assistant.
Use the context below to answer accurately and concisely.

Context:
{ctx}

Question:
{question}
""")

    prompt_val = prompt.format_prompt(ctx=combined_text, question=question)
    return generate_llm_answer(prompt_val)

# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    print("[INFO] Loading campaign JSON files…")
    campaigns = load_campaigns(DATA_FOLDER)

    print("[INFO] Creating LangChain documents…")
    documents = [create_document(c) for c in campaigns]

    print("[INFO] Splitting into text chunks…")
    chunked_docs = split_documents(documents)

    print("[INFO] Building FAISS vectorstore…")
    vectorstore, embeddings = build_vectorstore(chunked_docs)

    print("[INFO] Saving numeric + text summaries…")
    save_summaries(campaigns)

    print("[INFO] Computing numeric aggregates…")
    numeric_aggregates = compute_numeric_aggregates(campaigns)

    print("\n[READY] RAG system is running!\n")

    # Interactive CLI
    while True:
        q = input("Ask a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        print("\nAnswer:\n", get_insight_from_rag(vectorstore, embeddings, chunked_docs, numeric_aggregates, q))
        print("-" * 60)
