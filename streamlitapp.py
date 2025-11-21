# -------------------------------
# app.py (Streamlit Hybrid RAG)
# -------------------------------

import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Import your hybrid RAG functions and config
from src.raghybrid import (
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
# 1. Load embeddings & FAISS (once)
# -------------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore, embeddings

vectorstore, embeddings = load_vectorstore()

# -------------------------------
# 2. Load campaigns & compute numeric aggregates (once)
# -------------------------------
@st.cache_data
def load_campaigns_and_aggregates():
    campaigns = load_campaigns(DATA_FOLDER)
    documents = [create_document(c) for c in campaigns]
    chunked_docs = split_documents(documents)

    df = pd.DataFrame(campaigns)
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

    return campaigns, documents, chunked_docs, numeric_aggregates

campaigns, documents, chunked_docs, numeric_aggregates = load_campaigns_and_aggregates()

# -------------------------------
# 3. Streamlit UI
# -------------------------------
st.set_page_config(page_title="Crowdfunding Hybrid RAG", layout="wide")
st.title("📊 Crowdfunding Hybrid RAG Assistant")

st.markdown(
    """
Ask numeric or semantic questions about crowdfunding campaigns.
Examples:
- Who has the maximum raised amount?
- Average supporters
- Who has the highest percentage of goal funded?
- Summarize campaigns about children
"""
)

query = st.text_input("Enter your question:", "")

if st.button("Get Answer") and query.strip():
    with st.spinner("Generating answer..."):
        answer = get_insight_from_rag(
            vectorstore=vectorstore,
            embeddings=embeddings,
            documents=chunked_docs,
            numeric_aggregates=numeric_aggregates,
            question=query,
            k=5
        )
    st.subheader("Answer:")
    st.write(answer)

st.markdown("---")
st.info("Hybrid RAG system: semantic + numeric insights.")
