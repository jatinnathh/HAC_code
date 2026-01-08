

import pathway as pw



# 1. Read as BINARY to grab the whole file at once

#    (Unlike plaintext, this does not split on new lines)

raw_data = pw.io.fs.read(

    path="/content/data/Books/",

    format="binary",

    mode="static",

    with_metadata=True

)



# 2. Decode the bytes into a String

#    We use a UDF to convert the raw bytes (b'Chapter 1...') into text.

@pw.udf

def decode_file(data: bytes) -> str:

    try:

        return data.decode('utf-8')

    except:

        return data.decode('latin-1') # Fallback for older books



@pw.udf

def as_string(val) -> str:

    return str(val)



documents = raw_data.select(

    text=decode_file(pw.this.data),

    doc_id=as_string(pw.this._metadata["path"])

)



# 3. VERIFICATION: Check the size now

#    (We convert to pandas just for this print statement)

df = pw.debug.table_to_pandas(documents)



print(f"--- READ COMPLETE ---")

print(f"Total Books Loaded: {len(df)}") # Should be 2 (one row per file)



if not df.empty:

    for index, row in df.iterrows():

        print(f"\nBook: {row['doc_id']}")

        print(f"Length: {len(row['text'])} characters") # Should be > 500,000

        print("Start of text:", row['text'][:100].replace('\n', ' '))



import pathway as pw



@pw.udf

def decode_binary(data: bytes) -> str:

    try:

        return data.decode("utf-8")

    except Exception:

        return data.decode("latin-1", errors="ignore")



import re

import pathway as pw



@pw.udf

def split_into_chapters(text: str) -> list[str]:

    """

    Split a novel into chapters using Gutenberg-style headers.

    """

    if not text:

        return []



    # Preserve paragraph structure

    text = re.sub(r"\n{3,}", "\n\n", text)



    # Split on CHAPTER headings

    parts = re.split(r"\n\s*(CHAPTER\s+[IVXLCDM0-9]+.*?)\n", text)



    chapters = []

    for i in range(1, len(parts), 2):

        chapter_text = parts[i] + "\n" + parts[i + 1]

        chapters.append(chapter_text)



    # Fallback: if no chapters detected, return whole book

    return chapters if chapters else [text]





from langchain_experimental.text_splitter import SemanticChunker

from langchain_community.embeddings import HuggingFaceEmbeddings



# Model ONLY for detecting semantic boundaries

semantic_embedder = HuggingFaceEmbeddings(

    model_name="all-MiniLM-L6-v2"

)



@pw.udf

def semantic_chunk_chapter(

    chapter_text: str,

    min_chars: int = 500,

    max_chars: int = 2500

) -> list[tuple[str, dict]]:

    """

    Semantic chunking with size guards.

    Produces scene-level chunks.

    """

    if not chapter_text:

        return []



    # IMPORTANT: preserve paragraphs

    chapter_text = re.sub(r"\n{3,}", "\n\n", chapter_text)



    splitter = SemanticChunker(

        semantic_embedder,

        breakpoint_threshold_type="percentile",

        breakpoint_threshold_amount=90

    )



    docs = splitter.create_documents([chapter_text])

    raw_chunks = [d.page_content for d in docs]



    # --- Size Guard: merge small semantic chunks ---

    final_chunks = []

    buffer = ""



    for chunk in raw_chunks:

        if len(buffer) + len(chunk) < min_chars:

            buffer += " " + chunk

        else:

            final_chunks.append(buffer.strip())

            buffer = chunk



    if buffer.strip():

        final_chunks.append(buffer.strip())



    # Hard cap (safety)

    bounded_chunks = []

    for c in final_chunks:

        if len(c) > max_chars:

            bounded_chunks.extend(

                [c[i:i+max_chars] for i in range(0, len(c), max_chars)]

            )

        else:

            bounded_chunks.append(c)



    return [

        (text, {"strategy": "chapter_semantic", "chunk_index": i})

        for i, text in enumerate(bounded_chunks)

    ]





def run_ingestion_pipeline(data_dir="/content/data/Books/"):

    # Read binary (safe for large novels)

    raw = pw.io.fs.read(

        data_dir,

        format="binary",

        mode="static",

        with_metadata=True

    )



    # Decode

    documents = raw.select(

        path=pw.this._metadata["path"],

        text=decode_binary(pw.this.data)

    )



    # Chapter segmentation

    chapters = documents.select(

        path=pw.this.path,

        chapters=split_into_chapters(pw.this.text)

    ).flatten(pw.this.chapters)



    # Semantic chunking per chapter

    chunked = chapters.select(

        path=pw.this.path,

        chunks=semantic_chunk_chapter(pw.this.chapters)

    )



    # Flatten to one-row-per-chunk

    final_chunks = chunked.flatten(pw.this.chunks).select(

        path=pw.this.path,

        chunk_text=pw.this.chunks[0],

        chunk_metadata=pw.this.chunks[1]

    )



    return final_chunks





pipeline = run_ingestion_pipeline()

df = pw.debug.table_to_pandas(pipeline)



df["char_len"] = df["chunk_text"].str.len()

df["char_len"].describe()





