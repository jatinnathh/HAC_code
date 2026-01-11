# -*- coding: utf-8 -*-
"""
Advanced Narrative Reasoning Pipeline
Strategy: Atomic Claim Decomposition + Aggregate NLI Constraints
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
        "scikit-learn",
        "nltk"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

try:
    import pathway as pw
    import torch
    import nltk
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    print("⏳ Installing required packages...")
    install_packages()
    print("✅ Installation complete.")
    import pathway as pw
    import torch
    import nltk
    from langchain_text_splitters import RecursiveCharacterTextSplitter

# === 🛠️ NLTK FIX: Download both 'punkt' and 'punkt_tab' ===
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("⏳ Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('punkt_tab')

import pandas as pd
import numpy as np
import re
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathway.stdlib.ml.index import KNNIndex
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on: {device}")

# Download Data
if not os.path.exists("data"):
    print("Downloading dataset...")
    subprocess.run(["git", "clone", "https://github.com/jatinnathh/data.git"], check=True)

# ==========================================
# 2. CONFIGURATION & UDFS
# ==========================================

# Increased chunk size to capture more "Constraint Context"
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,      
    chunk_overlap=100,
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
    if not text: return []
    return text_splitter.split_text(text)

@pw.udf
def embed_text(text: str) -> list[float]:
    if not text: return [0.0] * 384
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
# 5. RETRIEVAL (DIVERSE SEARCH)
# ==========================================

# Fetch 80 candidates to ensure we cover multiple parts of the book
TOP_K = 80

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

# --- ROBUST MATCHING ---
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

# Group chunks
grouped_evidence = df_filtered.groupby('question_id')['chunk_text'].apply(list).reset_index()
grouped_queries = df_filtered[['question_id', 'backstory', 'label']].drop_duplicates()
final_dataset = pd.merge(grouped_queries, grouped_evidence, on='question_id', how='left')

print(f"📊 Ready for Reasoning. Rows: {len(final_dataset)}")

# ==========================================
# 6. ADVANCED REASONING MODULE
# ==========================================

if final_dataset.empty:
    print("❌ ERROR: No matching data found.")
else:
    print("⏳ Loading NLI Model (DeBERTa-v3-base)...")
    # DeBERTa-v3-base is the best balance of reasoning and speed
    nli_model_name = "cross-encoder/nli-deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)

    def extract_atomic_claims(backstory):
        """Splits backstory into atomic claims for granular verification."""
        try:
            sentences = nltk.tokenize.sent_tokenize(str(backstory))
            # Filter out very short sentences
            return [s for s in sentences if len(s) > 15]
        except Exception as e:
            # Fallback if NLTK fails for any reason
            return [str(backstory)]

    def perform_reasoning(backstory, evidence_chunks):
        if not evidence_chunks or not isinstance(evidence_chunks, list) or len(evidence_chunks) == 0:
            return 1, "Insufficient evidence to verify claims."
        
        # 1. Decompose Backstory into Claims
        claims = extract_atomic_claims(backstory)
        if not claims: claims = [str(backstory)] # Fallback
        
        # 2. Select Diverse Evidence (Top 15 to cover "multiple parts")
        cleaned_evidence = []
        for c in evidence_chunks[:15]: 
            cleaned_evidence.append(str(c[0]) if isinstance(c, (tuple, list)) else str(c))
            
        global_contradictions = []

        # 3. Check EACH claim against ALL evidence
        for claim in claims:
            # Prepare batch: (Claim, Chunk 1), (Claim, Chunk 2)...
            pairs = [[claim, chunk] for chunk in cleaned_evidence]
            
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
            
            with torch.no_grad():
                scores = nli_model(**inputs).logits
            
            probs = torch.softmax(scores, dim=1)
            contra_scores = probs[:, 0].cpu().numpy() # Index 0 = Contradiction
            
            # Find the strongest contradiction for THIS claim
            max_idx = np.argmax(contra_scores)
            max_score = contra_scores[max_idx]
            
            if max_score > 0.5: # Capture even weak signals for aggregation
                global_contradictions.append({
                    "claim": claim,
                    "evidence": cleaned_evidence[max_idx],
                    "score": max_score
                })
        
        # 4. CONSTRAINT LOGIC
        
        # Sort contradictions by severity
        global_contradictions.sort(key=lambda x: x['score'], reverse=True)
        
        # Rule A: Single Fatal Contradiction (The "Impossible" Event)
        # If we are >95% sure one specific claim is false, the whole backstory is false.
        if len(global_contradictions) > 0 and global_contradictions[0]['score'] > 0.95:
             top_case = global_contradictions[0]
             rationale = (f"The narrative constraint is violated. The claim '{top_case['claim']}' "
                          f"is directly contradicted by the text: '{top_case['evidence'][:150]}...' "
                          f"(Confidence: {top_case['score']:.2f}).")
             return 0, rationale

        # Rule B: Accumulation of Evidence (The "Constraint" Check)
        # If we have 2+ independent claims that are moderately contradicted (> 0.75),
        # it implies the character's *archetype* fits poorly.
        high_conf_conflicts = [c for c in global_contradictions if c['score'] > 0.75]
        
        if len(high_conf_conflicts) >= 2:
            c1 = high_conf_conflicts[0]
            c2 = high_conf_conflicts[1]
            rationale = (f"Multiple narrative inconsistencies found. "
                         f"Evidence suggests '{c1['evidence'][:60]}...' refutes '{c1['claim']}', "
                         f"and '{c2['evidence'][:60]}...' refutes '{c2['claim']}'.")
            return 0, rationale

        # Rule C: Consistent
        return 1, "The narrative events and character background align with the text constraints."

    print("⏳ Running Advanced Constraint Reasoning...")
    
    preds = []
    rationales = []
    
    for idx, row in final_dataset.iterrows():
        label, reason = perform_reasoning(row['backstory'], row['chunk_text'])
        preds.append(label)
        rationales.append(reason)
        
        if idx % 10 == 0: print(f"Reasoning over Q {idx}...", end="\r")

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
    
    print("\n\n🎉 FINAL RESULTS (Reasoning Pipeline) 🎉")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(final_dataset['y_true'], final_dataset['prediction']))
    
    final_dataset.to_csv("submission_reasoning.csv", index=False)
    print("✅ Results saved.")
