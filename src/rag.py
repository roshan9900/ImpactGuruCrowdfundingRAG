# -------------------------------
# crowdfunding_rag.py
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

# Updated HuggingFace embeddings import
from langchain_huggingface import HuggingFaceEmbeddings

# Optional: Groq LLM (requires API key)
from langchain_groq import ChatGroq

load_dotenv()

# -------------------------------
# CONFIG
# -------------------------------
DATA_FOLDER = "docs_json"           # Folder containing JSON files
VECTORSTORE_PATH = "faiss_index"    # FAISS index save path
NUMERIC_SUMMARY_FILE = "numeric_summary.csv"
TEXT_SUMMARY_FILE = "text_summary.txt"
OVERALL_SUMMARY_FILE = "overall_summary.txt"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"

# -------------------------------
# 1. LOAD DATA FROM JSON
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
# 2. CREATE DOCUMENTS WITH METADATA
# -------------------------------
def create_document(campaign):
    content = campaign.get("Story Summary", "") or "No story provided."
    metadata = {
        "url": campaign.get("Campaign URL", ""),
        "campaigner": campaign.get("Campaigner", ""),
        "age_group": campaign.get("Age Group", ""),
        "raised": campaign.get("Raised", 0.0),
        "goal": campaign.get("Goal", 0.0),
        "supporters": campaign.get("Supporters", 0),
        "avg_donation": campaign.get("Avg Donation", 0.0),
        "days_left": campaign.get("Days Left", 0),
        "urgency": campaign.get("Urgency", ""),
        "funding_gap": campaign.get("Funding Gap", ""),
        "sentiment": campaign.get("Sentiment", "")
    }
    return Document(page_content=content, metadata=metadata)

# -------------------------------
# 3. SPLIT TEXT INTO CHUNKS
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
# 4. CREATE EMBEDDINGS AND FAISS VECTORSTORE
# -------------------------------
def build_vectorstore(documents):
    valid_docs = [doc for doc in documents if doc.page_content.strip()]
    if not valid_docs:
        raise ValueError("No valid documents to embed.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(valid_docs, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore, embeddings  # Return embeddings separately

# -------------------------------
# 5. SAVE SUMMARIES
# -------------------------------
def save_summaries(campaigns):
    if not campaigns:
        print("[WARN] No campaigns to summarize")
        return

    df = pd.DataFrame(campaigns)

    # Numeric summary
    numeric_cols = ["Raised", "Goal", "Supporters", "Days Left", "Avg Donation"]
    numeric_summary = df[numeric_cols].describe()
    numeric_summary.to_csv(NUMERIC_SUMMARY_FILE)

    # Text summary
    with open(TEXT_SUMMARY_FILE, "w", encoding="utf-8") as f:
        for camp in campaigns:
            story = camp.get("Story Summary", "")
            if story:
                f.write(story + "\n\n")

    # Overall summary
    overall = f"Total campaigns: {len(campaigns)}\n"
    total_raised = sum([c.get("Raised", 0.0) for c in campaigns])
    total_goal = sum([c.get("Goal", 0.0) for c in campaigns])
    overall += f"Total raised: {total_raised}\nTotal goal: {total_goal}\n"
    with open(OVERALL_SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(overall)

# -------------------------------
# 6. QUERY FUNCTION (RAG)
# -------------------------------
def get_insight_from_rag(vectorstore, embeddings, documents, question, k=5, metadata_filter=None):
    search_docs = vectorstore

    if metadata_filter:
        print(f"[INFO] Applying filter: {metadata_filter}")
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

    # Initialize LLM
    llm = ChatGroq(model=LLM_MODEL)

    # Prompt template
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

    print("[INFO] RAG system ready!")
    print("You can query with:")
    print("get_insight_from_rag(vectorstore, embeddings, chunked_docs, 'Your question', k=5, metadata_filter={'age_group':'Child'})")

    # Interactive CLI
    while True:
        q = input("Ask a question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        a = get_insight_from_rag(vectorstore, embeddings, chunked_docs, q)
        print("\nAnswer:\n", a)
        print("-"*50)
