import pandas as pd
import numpy as np
from textblob import TextBlob
from tqdm import tqdm
import os
import json

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(r"C:\Users\hp\Documents\Impactguru\ketto_campaigns.csv")
df = df.astype(str).fillna("")

# Convert numeric columns safely
numeric_cols = ["raised_amount", "goal_amount", "supporters", "days_left"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# ===============================
# HELPER FUNCTIONS
# ===============================

def compute_funding_percent(raised, goal):
    if goal > 0:
        return round((raised / goal) * 100, 2)
    return 0

def avg_donation_per_supporter(raised, supporters):
    if supporters > 0:
        return round(raised / supporters, 2)
    return 0

def urgency_level(days_left, funding_percent):
    if days_left <= 7 and funding_percent < 50:
        return "High"
    elif days_left <= 15 and funding_percent < 70:
        return "Medium"
    else:
        return "Low"

def funding_gap_classification(funding_percent):
    if funding_percent < 25:
        return "High Gap"
    elif funding_percent < 75:
        return "Medium Gap"
    else:
        return "Low Gap"

def sentiment_score(text):
    if not text.strip():
        return "Neutral"
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.3:
        return "Positive"
    elif polarity < -0.2:
        return "Negative"
    else:
        return "Neutral"

def story_summary(text):
    if not text.strip():
        return "No story available."
    summary = text.replace("\n", " ").strip()
    return summary

def categorize_age_relation(text):
    text_lower = text.lower()
    if "daughter" in text_lower or "son" in text_lower or "child" in text_lower or "baby" in text_lower or "twins" in text_lower:
        return "Child"
    elif "mother" in text_lower or "father" in text_lower or "aunt" in text_lower or "uncle" in text_lower:
        return "Adult"
    else:
        return "Adult"

# ===============================
# GENERATE ENRICHED DOCUMENTS
# ===============================
enriched_documents = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing campaigns"):
    raised = row["raised_amount"]
    goal = row["goal_amount"]
    supporters = row["supporters"]
    days = row["days_left"]
    story = row["story"]

    funding_percent = compute_funding_percent(raised, goal)
    avg_donation = avg_donation_per_supporter(raised, supporters)
    urgency = urgency_level(days, funding_percent)
    gap_class = funding_gap_classification(funding_percent)
    sentiment = sentiment_score(story)
    summary = story_summary(story)
    age_group = categorize_age_relation(story)

    # Create JSON-friendly dict
    campaign_dict = {
        "Campaign URL": row.get("url", ""),
        "Campaigner": row.get("campaigner_name", ""),
        "Age Group": age_group,
        "Raised": raised,
        "Goal": goal,
        "Funding Percent": funding_percent,
        "Supporters": supporters,
        "Avg Donation": avg_donation,
        "Days Left": days,
        "Urgency": urgency,
        "Funding Gap": gap_class,
        "Sentiment": sentiment,
        "Story Summary": summary
    }

    enriched_documents.append(campaign_dict)

# ===============================
# SAVE DOCUMENTS AS JSON FILES
# ===============================
os.makedirs('docs_json', exist_ok=True)
for i, doc in enumerate(enriched_documents):
    filename = f"docs_json/campaign_doc_{i+1}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=4)

print(f"[INFO] {len(enriched_documents)} campaign JSON documents saved in 'docs_json' folder.")

# ===============================
# OPTIONAL: SAVE ENRICHED CSV
# ===============================
df["funding_percent"] = df.apply(lambda row: compute_funding_percent(row["raised_amount"], row["goal_amount"]), axis=1)
df["avg_donation"] = df.apply(lambda row: avg_donation_per_supporter(row["raised_amount"], row["supporters"]), axis=1)
df["urgency"] = df.apply(lambda row: urgency_level(row["days_left"], row["funding_percent"]), axis=1)
df["funding_gap_class"] = df["funding_percent"].apply(funding_gap_classification)
df["sentiment"] = df["story"].apply(sentiment_score)
df["story_summary"] = df["story"].apply(story_summary)
df["age_group"] = df["story"].apply(categorize_age_relation)

df.to_csv("ketto_campaigns_enriched.csv", index=False)
print("[INFO] Enriched CSV saved → ketto_campaigns_enriched.csv")
