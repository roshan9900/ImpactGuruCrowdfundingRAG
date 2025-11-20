# -------------------------------
# infer.py
# -------------------------------

import os
import json
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Import your hybrid RAG function and config
from raghybrid import (
    get_insight_from_rag,
    VECTORSTORE_PATH,
    EMBEDDING_MODEL,
    DATA_FOLDER,
    NUMERIC_COLUMNS,
    load_campaigns,
    create_document,
    split_documents
)

# -------------------------------
# 1. Load embeddings & FAISS
# -------------------------------
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# -------------------------------
# 2. Load campaigns & documents
# -------------------------------
print("[INFO] Loading campaigns for numeric aggregates...")
campaigns = load_campaigns(DATA_FOLDER)
documents = [create_document(c) for c in campaigns]
chunked_docs = split_documents(documents)

# -------------------------------
# 3. Compute numeric aggregates
# -------------------------------
df = pd.DataFrame(campaigns)

# Add derived column: Raised % of Goal
df["Raised % of Goal"] = df.apply(lambda x: (x["Raised"]/x["Goal"]*100) if x["Goal"] else 0, axis=1)

numeric_aggregates = {}
for col in NUMERIC_COLUMNS + ["Raised % of Goal"]:
    if col in df.columns:
        numeric_aggregates[col] = {
            "sum": df[col].sum(),
            "mean": df[col].mean(),
            "max": df[col].max(),
            "min": df[col].min(),
            "count": df[col].count(),
            "df": df
        }

# -------------------------------
# 4. Interactive Query Loop
# -------------------------------
print("[INFO] Hybrid RAG ready! You can ask numeric or semantic questions.")
print("Type 'exit' to quit.")

while True:
    query = input("\nAsk a question: ")
    if query.lower() == "exit":
        break

    answer = get_insight_from_rag(
        vectorstore=vectorstore,
        embeddings=embeddings,
        documents=chunked_docs,
        numeric_aggregates=numeric_aggregates,
        question=query,
        k=5
    )

    print("\nAnswer:\n", answer)
    print("-"*50)
