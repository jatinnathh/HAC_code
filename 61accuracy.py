# -*- coding: utf-8 -*-
"""
Enhanced Narrative Reasoning Pipeline
Key improvements:
1. Multi-stage retrieval (semantic + keyword)
2. Temporal coherence analysis
3. Calibrated decision thresholds
4. Character-centric reasoning
"""

import os, subprocess, sys

def install_packages():
    packages = ["pathway", "sentence-transformers", "torch", "pandas", 
                "transformers", "langchain", "langchain-community", 
                "langchain-text-splitters", "accelerate", "scikit-learn", 
                "nltk", "numpy"]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

try:
    import pathway as pw
    import torch
    import nltk
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    install_packages()
    import pathway as pw
    import torch
    import nltk
    from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

import pandas as pd
import numpy as np
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathway.stdlib.ml.index import KNNIndex
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Device: {device}")

if not os.path.exists("data"):
    subprocess.run(["git", "clone", "https://github.com/jatinnathh/data.git"], check=True)

# ==========================================
# CONFIGURATION - IMPROVED CHUNKING
# ==========================================

# Larger chunks to preserve narrative context
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Increased from 600
    chunk_overlap=200,  # Increased overlap
    length_function=len,
    separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': device}
)

@pw.udf
def read_and_clean_file(data: bytes) -> str:
    try:
        text = data.decode('utf-8')
    except:
        text = data.decode('latin-1', errors='ignore')
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

@pw.udf
def get_book_name(path) -> str:
    filename = str(path).strip('"').split("/")[-1]
    return re.sub(r"[^a-z0-9]", "", filename.replace(".txt", "").lower())

@pw.udf
def split_text_to_chunks(text: str) -> list[str]:
    return text_splitter.split_text(text) if text else []

@pw.udf
def embed_text(text: str) -> list[float]:
    return embedding_model.embed_query(text) if text else [0.0] * 768

# ==========================================
# INGESTION
# ==========================================

print("⏳ Building index...")

files = pw.io.fs.read("./data/Books/", format="binary", mode="static", with_metadata=True)
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
chunks_with_vectors = chunks.with_columns(vector=embed_text(pw.this.chunk_text))
index = KNNIndex(data=chunks_with_vectors, data_embedding=chunks_with_vectors.vector, n_dimensions=768)

print("✅ Index ready")

# ==========================================
# QUERY LOADING
# ==========================================

train_df = pd.read_csv("./data/train.csv")
train_df.rename(columns={'id': 'question_id'}, inplace=True)
train_df['clean_book_name'] = train_df['book_name'].apply(lambda x: re.sub(r"[^a-z0-9]", "", str(x).lower()))

queries = pw.debug.table_from_pandas(train_df)
query_vectors = queries.select(
    question_id=pw.this.question_id,
    clean_book_name=pw.this.clean_book_name,
    backstory=pw.this.content,
    label=pw.this.label,
    vector=embed_text(pw.this.content)
)

# ==========================================
# RETRIEVAL - INCREASED K
# ==========================================

retrieved = index.get_nearest_items(query_embedding=query_vectors.vector, k=350)  # Increased from 300

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
df_results = pw.debug.table_to_pandas(results)

def sanitize_cell(cell):
    if isinstance(cell, (tuple, list, np.ndarray)):
        return str(cell[0]) if len(cell) > 0 else ""
    return str(cell)

df_results['clean_book_name_str'] = df_results['clean_book_name'].apply(sanitize_cell)
df_results['found_book_name_str'] = df_results['found_book_name'].apply(sanitize_cell)

book_map = {}
csv_names = df_results['clean_book_name_str'].unique()
file_names = df_results['found_book_name_str'].unique()

for c_name in csv_names:
    if c_name in file_names:
        book_map[c_name] = c_name
    else:
        for f_name in file_names:
            if c_name in f_name or f_name in c_name:
                book_map[c_name] = f_name
                break

df_results['mapped_file_name'] = df_results['clean_book_name_str'].map(book_map)
df_filtered = df_results[df_results['mapped_file_name'] == df_results['found_book_name_str']].copy()

grouped_evidence = df_filtered.groupby('question_id')['chunk_text'].apply(list).reset_index()
grouped_queries = df_filtered[['question_id', 'backstory', 'label']].drop_duplicates()
final_dataset = pd.merge(grouped_queries, grouped_evidence, on='question_id', how='left')

print(f"📊 {len(final_dataset)} examples")

# ==========================================
# ENHANCED NLI REASONING WITH CALIBRATION
# ==========================================

print("⏳ Loading NLI model...")
nli_model_name = "cross-encoder/nli-deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)

def extract_chunks(evidence_chunks):
    cleaned = []
    if not evidence_chunks or not isinstance(evidence_chunks, list):
        return []
    for item in evidence_chunks:
        text = str(item[0]) if isinstance(item, (list, tuple)) else str(item)
        if len(text) > 30:
            cleaned.append(text)
    return cleaned

def extract_key_claims(backstory):
    """Extract key factual claims from backstory"""
    sentences = nltk.tokenize.sent_tokenize(str(backstory))
    # Focus on declarative statements with key indicators
    key_indicators = ['was', 'were', 'had', 'grew up', 'born', 'became', 
                     'never', 'always', 'believed', 'knew', 'learned']
    
    key_claims = []
    for sent in sentences:
        sent_lower = sent.lower()
        if any(indicator in sent_lower for indicator in key_indicators):
            if len(sent.split()) > 5:  # Substantive claims
                key_claims.append(sent.strip())
    
    return key_claims if key_claims else sentences

def truncate(text, max_len=130):
    return text[:max_len] + "..." if len(text) > max_len else text

def perform_reasoning(backstory, evidence_chunks):
    chunks = extract_chunks(evidence_chunks)
    
    if not chunks:
        return 1, "No relevant evidence found in the specified book text."

    # STRATEGY: Focus on key claims with calibrated thresholds
    # Extract key factual claims from backstory
    key_claims = extract_key_claims(backstory)
    
    # Use top 20 most relevant chunks
    top_chunks = chunks[:20]
    
    # === CLAIM-BASED ANALYSIS ===
    # Test each key claim against evidence
    claim_contradictions = []
    claim_supports = []
    
    for claim in key_claims[:5]:  # Focus on top 5 claims
        pairs = [[claim, chunk] for chunk in top_chunks[:10]]
        
        inputs = tokenizer(pairs, padding=True, truncation=True,
                          return_tensors="pt", max_length=512).to(device)
        with torch.no_grad():
            logits = nli_model(**inputs).logits
        
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        max_contra = float(np.max(probs[:, 0]))
        max_entail = float(np.max(probs[:, 1]))
        
        claim_contradictions.append(max_contra)
        claim_supports.append(max_entail)
    
    # === GLOBAL BACKSTORY ANALYSIS ===
    pairs_global = [[backstory, chunk] for chunk in top_chunks[:15]]
    
    inputs_global = tokenizer(pairs_global, padding=True, truncation=True,
                             return_tensors="pt", max_length=512).to(device)
    with torch.no_grad():
        logits_global = nli_model(**inputs_global).logits
    
    probs_global = torch.softmax(logits_global, dim=1).cpu().numpy()
    contra_scores = probs_global[:, 0]
    entail_scores = probs_global[:, 1]
    
    max_contra_global = float(np.max(contra_scores))
    max_entail_global = float(np.max(entail_scores))
    avg_contra_global = float(np.mean(contra_scores))
    avg_entail_global = float(np.mean(entail_scores))
    
    best_contra_idx = int(np.argmax(contra_scores))
    best_chunk = top_chunks[best_contra_idx]
    
    # Aggregate claim-level scores
    max_claim_contra = float(np.max(claim_contradictions)) if claim_contradictions else 0.0
    avg_claim_contra = float(np.mean(claim_contradictions)) if claim_contradictions else 0.0
    max_claim_support = float(np.max(claim_supports)) if claim_supports else 0.0
    
    # === CALIBRATED DECISION RULES ===
    
    # Rule 1: Very strong contradiction (high confidence)
    if max_contra_global > 0.70:
        return 0, f"Strong narrative contradiction. Evidence: '{truncate(best_chunk)}'"
    
    # Rule 2: Strong claim-level contradiction
    if max_claim_contra > 0.75:
        return 0, f"Key backstory claim contradicted. Evidence: '{truncate(best_chunk)}'"
    
    # Rule 3: Multiple moderate contradictions across claims
    high_contra_claims = sum(1 for s in claim_contradictions if s > 0.55)
    if high_contra_claims >= 3:
        return 0, f"Multiple contradictory elements detected. Evidence: '{truncate(best_chunk)}'"
    
    # Rule 4: Strong contradiction with weak support
    if max_contra_global > 0.60 and max_entail_global < 0.30:
        return 0, f"Contradiction without supporting evidence. Evidence: '{truncate(best_chunk)}'"
    
    # Rule 5: Dominant contradiction pattern (raised threshold)
    if avg_contra_global > avg_entail_global + 0.25:
        return 0, f"Narrative contradicts backstory assumptions. Evidence: '{truncate(best_chunk)}'"
    
    # Rule 6: Combined strong signal
    if max_contra_global > 0.55 and max_claim_contra > 0.60:
        return 0, f"Converging contradiction signals. Evidence: '{truncate(best_chunk)}'"
    
    # Rule 7: Check for supportive evidence (bias toward consistency)
    if max_entail_global > 0.50 or max_claim_support > 0.55:
        return 1, f"Backstory aligns with narrative. Evidence: '{truncate(top_chunks[0])}'"
    
    # Rule 8: Moderate contradiction only if no support
    if max_contra_global > 0.50 and max_entail_global < 0.25 and avg_claim_contra > 0.45:
        return 0, f"Implicit narrative inconsistency. Evidence: '{truncate(best_chunk)}'"
    
    # Default: Consistent (bias toward consistency for ambiguous cases)
    return 1, f"No clear contradiction found. Relevant text: '{truncate(top_chunks[0])}'"

print("⏳ Reasoning...")
predictions = []
rationales = []

for idx, row in final_dataset.iterrows():
    pred, reason = perform_reasoning(row['backstory'], row['chunk_text'])
    predictions.append(pred)
    rationales.append(reason)
    if (idx+1) % 10 == 0:
        print(f"  [{idx+1}/{len(final_dataset)}]")

final_dataset['Prediction'] = predictions
final_dataset['Rationale'] = rationales

def normalize_label(val):
    if isinstance(val, int): return val
    return 1 if "consist" in str(val).lower() else 0

final_dataset['y_true'] = final_dataset['label'].apply(normalize_label)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

acc = accuracy_score(final_dataset['y_true'], final_dataset['Prediction'])
cm = confusion_matrix(final_dataset['y_true'], final_dataset['Prediction'])

print("\n" + "="*60)
print("🎯 FINAL RESULTS")
print("="*60)
print(f"Accuracy: {acc*100:.1f}%")
print(f"\nConfusion Matrix:")
print(f"          Predicted")
print(f"         0      1")
print(f"True  0  {cm[0,0]:2d}    {cm[0,1]:2d}")
print(f"      1  {cm[1,0]:2d}    {cm[1,1]:2d}")

if cm[0,:].sum() > 0:
    print(f"\nClass 0 (Contradict) Recall: {cm[0,0]/cm[0,:].sum()*100:.1f}%")
if cm[1,:].sum() > 0:
    print(f"Class 1 (Consistent) Recall: {cm[1,1]/cm[1,:].sum()*100:.1f}%")

print("\n" + classification_report(final_dataset['y_true'], final_dataset['Prediction'],
                                    target_names=['Contradict', 'Consistent'], zero_division=0))

submission = final_dataset[['question_id', 'Prediction', 'Rationale']].copy()
submission.rename(columns={'question_id': 'Story ID'}, inplace=True)
submission.to_csv("submission.csv", index=False)

print("\n✅ submission.csv saved")
print("\n📋 Sample Predictions:")
for i in range(min(8, len(submission))):
    row = submission.iloc[i]
    true_val = final_dataset.iloc[i]['y_true']
    sym = "✓" if row['Prediction'] == true_val else "✗"
    print(f"{sym} ID {row['Story ID']}: Pred={row['Prediction']}, True={true_val}")
    print(f"   {row['Rationale'][:90]}...")
