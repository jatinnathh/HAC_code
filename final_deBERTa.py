# -*- coding: utf-8 -*-
"""
High-Accuracy NLP Pipeline for Narrative Consistency + Rationale Generation
Author: Gemini (Refined for Pathway + NLI + Rationale)
"""

# ==========================================
# 1. INSTALLATION & SETUP
# ==========================================
import os
import subprocess
import sys

def install_packages():
    packages = [
        "pathway",
        "sentence-transformers",
        "torch",
        "pandas",
        "transformers",
        "langchain",
        "langchain-community",
        "langchain-text-splitters",
        "accelerate",
        "scikit-learn"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

try:
    import pathway as pw
    import torch
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    print("⏳ Installing required packages...")
    install_packages()
    print("✅ Installation complete.")
    import pathway as pw
    import torch
    from langchain_text_splitters import RecursiveCharacterTextSplitter

import pandas as pd
import numpy as np
import re
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathway.stdlib.ml.index import KNNIndex
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on: {device}")

if not os.path.exists("data"):
    print("Downloading dataset...")
    subprocess.run(["git", "clone", "https://github.com/jatinnathh/data.git"], check=True)

# ==========================================
# 2. CONFIGURATION & UDFS
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ".", " ", ""]
)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@pw.udf
def read_and_clean_file(data: bytes) -> str:
    try:
        text = data.decode('utf-8')
    except:
        text = data.decode('latin-1', errors='ignore')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@pw.udf
def get_book_name(path) -> str:
    path_str = str(path).strip('"')
    filename = path_str.split("/")[-1]
    name = filename.replace(".txt", "").lower()
    name = re.sub(r"[^a-z0-9]", "", name) 
    return name

@pw.udf
def split_text_to_chunks(text: str) -> list[str]:
    if not text:
        return []
    return text_splitter.split_text(text)

@pw.udf
def embed_text(text: str) -> list[float]:
    if not text:
        return [0.0] * 384
    return embedding_model.embed_query(text)

# ==========================================
# 3. INGESTION PIPELINE
# ==========================================

print("⏳ Starting Ingestion Pipeline...")

files = pw.io.fs.read(
    path="./data/Books/",
    format="binary",
    mode="static",
    with_metadata=True
)

documents = files.select(
    path=pw.this._metadata["path"],
    book_name=get_book_name(pw.this._metadata["path"]),
    raw_text=read_and_clean_file(pw.this.data)
)

chunks = documents.select(
    book_name=pw.this.book_name,
    chunk_list=split_text_to_chunks(pw.this.raw_text)
).flatten(pw.this.chunk_list).select(
    book_name=pw.this.book_name,
    chunk_text=pw.this.chunk_list
)

chunks_with_vectors = chunks.with_columns(
    vector=embed_text(pw.this.chunk_text)
)

index = KNNIndex(
    data=chunks_with_vectors,
    data_embedding=chunks_with_vectors.vector,
    n_dimensions=384
)

print("✅ Index built successfully.")

# ==========================================
# 4. QUERY LOADING
# ==========================================

train_df = pd.read_csv("./data/train.csv")
train_df.rename(columns={'id': 'question_id'}, inplace=True)

def clean_csv_book_name(name):
    s = str(name).lower()
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

train_df['clean_book_name'] = train_df['book_name'].apply(clean_csv_book_name)

queries = pw.debug.table_from_pandas(train_df)

query_vectors = queries.select(
    question_id=pw.this.question_id,
    clean_book_name=pw.this.clean_book_name,
    backstory=pw.this.content,
    label=pw.this.label,
    vector=embed_text(pw.this.content)
)

# ==========================================
# 5. RETRIEVAL & AUTO-MATCHING
# ==========================================

TOP_K = 60

retrieved = index.get_nearest_items(
    query_embedding=query_vectors.vector,
    k=TOP_K
)

query_meta = query_vectors.select(
    question_id=pw.this.question_id,
    clean_book_name=pw.this.clean_book_name,
    backstory=pw.this.backstory,
    label=pw.this.label
)

retrieved_data = retrieved.select(
    chunk_text=pw.this.chunk_text,
    found_book_name=pw.this.book_name
)

results = query_meta + retrieved_data

print("⏳ Retrieving candidates...")
df_results = pw.debug.table_to_pandas(results)

# --- AUTO-MATCHING & CLEANUP ---
def sanitize_cell(cell):
    if isinstance(cell, (tuple, list, np.ndarray)):
        return str(cell[0]) if len(cell) > 0 else ""
    return str(cell)

df_results['clean_book_name_str'] = df_results['clean_book_name'].apply(sanitize_cell)
df_results['found_book_name_str'] = df_results['found_book_name'].apply(sanitize_cell)

csv_names = df_results['clean_book_name_str'].unique()
file_names = df_results['found_book_name_str'].unique()

book_map = {}
for c_name in csv_names:
    match_found = False
    if c_name in file_names:
        book_map[c_name] = c_name
        match_found = True
    else:
        for f_name in file_names:
            if c_name in f_name or f_name in c_name:
                book_map[c_name] = f_name
                match_found = True
                break
    if not match_found:
        print(f"⚠️ Warning: No file match found for book '{c_name}'")

df_results['mapped_file_name'] = df_results['clean_book_name_str'].map(book_map)
df_filtered = df_results[df_results['mapped_file_name'] == df_results['found_book_name_str']].copy()

print(f"📊 Rows after filtering: {len(df_filtered)}")

grouped_evidence = df_filtered.groupby('question_id')['chunk_text'].apply(list).reset_index()
grouped_queries = df_filtered[['question_id', 'backstory', 'label', 'clean_book_name_str']].drop_duplicates()
final_dataset = pd.merge(grouped_queries, grouped_evidence, on='question_id', how='left')

# ==========================================
# 6. PREDICTION & RATIONALE GENERATION
# ==========================================

if final_dataset.empty:
    print("❌ ERROR: No matching data found.")
else:
    print("⏳ Loading NLI Model (DeBERTa-v3-base)...")
    nli_model_name = "cross-encoder/nli-deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)

    def predict_and_explain(backstory, evidence_chunks):
        # Default if no data
        if not evidence_chunks or not isinstance(evidence_chunks, list) or len(evidence_chunks) == 0:
            return 1, "Insufficient evidence found to contradict the backstory."
        
        # Take top 5
        top_chunks = evidence_chunks[:5]
        
        # Sanitize Strings
        clean_pairs = []
        clean_chunk_texts = []
        for chunk in top_chunks:
            txt = str(chunk[0]) if isinstance(chunk, (tuple, list)) else str(chunk)
            bs = str(backstory)
            clean_pairs.append([bs, txt])
            clean_chunk_texts.append(txt)

        # Inference
        inputs = tokenizer(clean_pairs, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        with torch.no_grad():
            scores = nli_model(**inputs).logits
        
        probs = torch.softmax(scores, dim=1)
        contradiction_scores = probs[:, 0].cpu().numpy()
        entailment_scores = probs[:, 1].cpu().numpy()
        
        # Find max scores
        max_contra_idx = np.argmax(contradiction_scores)
        max_contra_score = contradiction_scores[max_contra_idx]
        
        # --- DECISION LOGIC ---
        
        # 1. If Strong Contradiction (> 0.90) -> Inconsistent (0)
        if max_contra_score > 0.90:
            evidence_text = clean_chunk_texts[max_contra_idx]
            rationale = f"Contradiction found (Score: {max_contra_score:.2f}). The narrative states: '{evidence_text[:200]}...'"
            return 0, rationale
            
        # 2. Otherwise -> Consistent (1)
        # We pick the chunk with highest entailment to show "proof" of consistency, 
        # or just the most relevant chunk if nothing entails strongly.
        best_support_idx = np.argmax(entailment_scores)
        evidence_text = clean_chunk_texts[best_support_idx]
        rationale = f"No strong contradiction found. Relevant narrative text: '{evidence_text[:200]}...'"
        return 1, rationale

    print("⏳ Running Inference & Generating Rationale...")
    
    preds = []
    rationales = []
    
    for idx, row in final_dataset.iterrows():
        label, reason = predict_and_explain(row['backstory'], row['chunk_text'])
        preds.append(label)
        rationales.append(reason)
        
        if idx % 10 == 0: print(f"Processing {idx}...", end="\r")

    final_dataset['prediction'] = preds
    final_dataset['rationale'] = rationales

    # Evaluation
    def normalize_label(val):
        if isinstance(val, int): return val
        return 1 if "consist" in str(val).lower() else 0

    final_dataset['y_true'] = final_dataset['label'].apply(normalize_label)

    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    
    acc = accuracy_score(final_dataset['y_true'], final_dataset['prediction'])
    cm = confusion_matrix(final_dataset['y_true'], final_dataset['prediction'])
    
    print("\n\n🎉 FINAL RESULTS 🎉")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(final_dataset['y_true'], final_dataset['prediction']))

    # Save output
    final_dataset.to_csv("submission_with_rationale.csv", index=False)
    print("✅ Saved to submission_with_rationale.csv")
