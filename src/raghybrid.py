# -------------------------------
# crowdfunding_rag_numeric.py
# -------------------------------

import os
import json
import pandas as pd
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# -------------------------------
# CONFIG
# -------------------------------
DATA_FOLDER = "docs_json"
VECTORSTORE_PATH = "faiss_index"
NUMERIC_SUMMARY_FILE = "numeric_summary.csv"
TEXT_SUMMARY_FILE = "text_summary.txt"
OVERALL_SUMMARY_FILE = "overall_summary.txt"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

NUMERIC_COLUMNS = ["Raised", "Goal", "Supporters", "Days Left", "Avg Donation"]

# -------------------------------
# 1. LOAD DATA
# -------------------------------
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

# -------------------------------
# 2. CREATE DOCUMENTS
# -------------------------------
def create_document(campaign):
    content = campaign.get("Story Summary", "") or "No story provided."
    metadata = {key: campaign.get(key, "") for key in ["Campaign URL","Campaigner","Age Group","Raised","Goal","Supporters","Avg Donation","Days Left","Urgency","Funding Gap","Sentiment"]}
    return Document(page_content=content, metadata=metadata)

# -------------------------------
# 3. SPLIT DOCUMENTS
# -------------------------------
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = []
    for doc in documents:
        if not doc.page_content.strip():
            continue
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata.copy()))
    return chunked_docs

# -------------------------------
# 4. BUILD VECTORSTORE
# -------------------------------
def build_vectorstore(documents):
    valid_docs = [doc for doc in documents if doc.page_content.strip()]
    if not valid_docs:
        raise ValueError("No valid documents to embed.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(valid_docs, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore, embeddings

# -------------------------------
# 5. SAVE SUMMARIES
# -------------------------------
def save_summaries(campaigns):
    if not campaigns:
        print("[WARN] No campaigns to summarize")
        return

    df = pd.DataFrame(campaigns)

    # Numeric summary
    numeric_summary = df[NUMERIC_COLUMNS].describe()
    numeric_summary.to_csv(NUMERIC_SUMMARY_FILE)

    # Text summary
    with open(TEXT_SUMMARY_FILE, "w", encoding="utf-8") as f:
        for camp in campaigns:
            story = camp.get("Story Summary", "")
            if story:
                f.write(story + "\n\n")

    # Overall summary
    overall = f"Total campaigns: {len(campaigns)}\n"
    total_raised = df["Raised"].sum()
    total_goal = df["Goal"].sum()
    overall += f"Total raised: {total_raised}\nTotal goal: {total_goal}\n"
    with open(OVERALL_SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(overall)

# -------------------------------
# 6. NUMERIC AGGREGATES
# -------------------------------
def compute_numeric_aggregates(campaigns):
    df = pd.DataFrame(campaigns)

    # Compute derived column: raised percentage of goal
    df["Raised % of Goal"] = df.apply(lambda x: (x["Raised"]/x["Goal"]*100) if x["Goal"] else 0, axis=1)

    aggregates = {}
    numeric_cols = NUMERIC_COLUMNS + ["Raised % of Goal"]

    for col in numeric_cols:
        if col in df.columns:
            aggregates[col] = {
                "sum": df[col].sum(),
                "mean": df[col].mean(),
                "max": df[col].max(),
                "min": df[col].min(),
                "count": df[col].count()
            }
            # Keep full DataFrame for derived queries
            aggregates[col]["df"] = df

    return aggregates


# -------------------------------
# 7. QUERY FUNCTION (RAG + Numeric)
# -------------------------------
def get_insight_from_rag(vectorstore, embeddings, documents, numeric_aggregates, question, k=5, metadata_filter=None):
    question_lower = question.lower()

    # Handle numeric/statistical questions
# Inside get_insight_from_rag function, replace numeric handling with this:

    if any(word in question_lower for word in ["average", "mean", "total", "sum", "max", "min", "highest", "lowest", "count", "percentage", "%"]):
        
        # Handle derived column: Raised % of Goal
        if "percentage" in question_lower or "% of goal" in question_lower:
            df = numeric_aggregates["Raised % of Goal"]["df"]
            max_row = df.loc[df["Raised % of Goal"].idxmax()]
            return f"Campaigner with max % raised: {max_row['Campaigner']} ({max_row['Raised % of Goal']:.2f}%)"

        # For all other numeric columns
        for col in numeric_aggregates.keys():
            agg = numeric_aggregates[col]
            df = agg.get("df")
            if col.lower() in question_lower:
                if "max" in question_lower or "highest" in question_lower:
                    row = df.loc[df[col].idxmax()]
                    return f"Campaigner with max {col}: {row['Campaigner']} ({row[col]:.2f})"
                elif "min" in question_lower or "lowest" in question_lower:
                    row = df.loc[df[col].idxmin()]
                    return f"Campaigner with min {col}: {row['Campaigner']} ({row[col]:.2f})"
                elif "average" in question_lower or "mean" in question_lower:
                    return f"Average {col}: {agg.get('mean', 'N/A'):.2f}"
                elif "total" in question_lower or "sum" in question_lower:
                    return f"Total {col}: {agg.get('sum', 'N/A'):.2f}"
                elif "count" in question_lower:
                    return f"Count for {col}: {agg.get('count', 'N/A')}"
        return "Could not identify numeric column in question."

    # Otherwise, use semantic search + LLM
    search_docs = vectorstore
    if metadata_filter:
        filtered_docs = [
            doc for doc in documents
            if all(str(doc.metadata.get(k, "")).lower() == str(v).lower()
                   for k, v in metadata_filter.items())
        ]
        if not filtered_docs:
            return "No campaigns match the filter criteria."
        search_docs = FAISS.from_documents(filtered_docs, embeddings)

    results = search_docs.similarity_search(question, k=k)
    if not results:
        return "No relevant campaigns found."

    campaigns_text = "\n\n".join([d.page_content for d in results])
    llm = ChatGroq(model=LLM_MODEL)
    prompt = ChatPromptTemplate.from_template("""
You are a crowdfunding assistant.
Given the following campaign information, answer the question concisely:

Campaigns:
{campaigns}

Question: {question}
""")
    prompt_value = prompt.format_prompt(campaigns=campaigns_text, question=question)
    answer_message = llm.invoke(prompt_value)
    return answer_message.content


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    print("[INFO] Loading campaigns...")
    campaigns = load_campaigns(DATA_FOLDER)

    print("[INFO] Creating documents...")
    documents = [create_document(c) for c in campaigns]

    print("[INFO] Splitting documents into chunks...")
    chunked_docs = split_documents(documents)

    print("[INFO] Building FAISS vectorstore...")
    vectorstore, embeddings = build_vectorstore(chunked_docs)

    print("[INFO] Saving summaries...")
    save_summaries(campaigns)

    print("[INFO] Computing numeric aggregates...")
    numeric_aggregates = compute_numeric_aggregates(campaigns)

    print("[INFO] RAG system ready! You can query numeric and text questions.")

    # Interactive CLI
    while True:
        q = input("Ask a question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        a = get_insight_from_rag(vectorstore, embeddings, chunked_docs, numeric_aggregates, q)
        print("\nAnswer:\n", a)
        print("-"*50)
