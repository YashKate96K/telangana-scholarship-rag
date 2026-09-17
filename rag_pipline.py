import re
import math
import time
from pathlib import Path
from collections import deque
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import fitz  # pymupdf
import tiktoken
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchAny
#from kaggle_secrets import UserSecretsClient
from groq import Groq
import atexit

# ---------------------------------------------------------------------------
# 1. Config / paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(r"D:\Downloads\RAG_PROJECT_TELANGANA_SS\data")
#PDF_PATH = Path(r"D:\Downloads\RAG_PROJECT_TELANGANA_SS\ilovepdf_merged.pdf")
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "scholarship_documents"

tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    return len(tokenizer.encode(text))


# ---------------------------------------------------------------------------
# 2. Load source documents
# ---------------------------------------------------------------------------
def load_csvs(data_dir):
    documents = []
    csv_files = sorted(data_dir.rglob("*.csv"))
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            for index, row in df.iterrows():
                text_parts = []
                for column in df.columns:
                    value = row[column]
                    if pd.notna(value):
                        text_parts.append(f"{column}: {value}")
                text = " | ".join(text_parts)
                if not text.strip():
                    continue
                documents.append({
                    "text": text,
                    "metadata": {
                        "source": csv_path.name,
                        "row": index + 1,
                        "file_type": "csv",
                        "raw_row": {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()},
                    },
                })
        except Exception as e:
            print(f"Error reading {csv_path.name}: {e}")
    return documents


def load_pdf(pdf_path, min_chars=40):
    if not pdf_path.exists():
        return []
    documents = []
    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if len(text) < min_chars:
                continue
            documents.append({
                "text": text,
                "metadata": {"source": pdf_path.name, "row": page_number, "file_type": "pdf", "raw_row": {}},
            })
    return documents


# ---------------------------------------------------------------------------
# 3. Clean
# ---------------------------------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*\|\s*", " | ", text)
    return text.strip()


def clean_documents(documents):
    cleaned, seen = [], set()
    for d in documents:
        text = clean_text(d["text"])
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append({"text": text, "metadata": d["metadata"]})
    return cleaned


# ---------------------------------------------------------------------------
# 4. Format documents + extract metadata
# ---------------------------------------------------------------------------
def clean_value(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def make_label(column_name):
    column_name = str(column_name).replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", column_name).strip().title()


def format_row_text(text):
    if not text:
        return ""
    formatted_fields = []
    for field in text.split("|"):
        field = field.strip()
        if not field:
            continue
        if ":" in field:
            column, value = field.split(":", 1)
            value = clean_value(value)
            if value:
                formatted_fields.append(f"{make_label(column)}: {value}")
        else:
            formatted_fields.append(field)
    return "\n".join(formatted_fields)


def detect_document_type(source):
    source = source.lower()
    mapping = [
        ("faqs_all_schemes", "FAQ"), ("faq", "FAQ"), ("eligibility", "Eligibility"),
        ("benefit", "Benefits"), ("hostel", "Hostel Information"), ("maintenance", "Maintenance Rates"),
        ("course", "Courses"), ("document", "Required Documents"), ("application", "Application Process"),
        ("rejection", "Rejection Rules"), ("timeline", "Service Timeline"), ("service", "Service Timeline"),
        ("contact", "Contacts and Grievances"), ("grievance", "Contacts and Grievances"),
        ("rti", "RTI Information"), ("citizen", "Citizen Charter"), ("scheme", "Scholarship Scheme"),
        ("source", "Source Information"), ("ilovepdf", "Official PDF Document"),
    ]
    for key, doc_type in mapping:
        if key in source:
            return doc_type
    return "General Scholarship Information"


def extract_field(text, field_names):
    for field in field_names:
        match = re.search(rf"^{re.escape(field)}:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    return None


def get_from_raw_row(raw_row, possible_columns):
    for name in possible_columns:
        if name in raw_row and raw_row[name] not in (None, ""):
            return str(raw_row[name]).strip()
    return None


def format_document(document):
    original_text = document.get("text", "")
    old_metadata = document.get("metadata", {})
    source = old_metadata.get("source", "unknown")
    row = old_metadata.get("row")
    raw_row = old_metadata.get("raw_row", {}) or {}
    file_type = old_metadata.get("file_type", "csv")

    formatted_text = format_row_text(original_text) if file_type == "csv" else original_text
    document_type = detect_document_type(source)

    scheme_id = get_from_raw_row(raw_row, ["scheme_id", "Scheme Id", "Scheme ID"]) \
        or extract_field(formatted_text, ["Scheme Id", "Scheme ID"])
    scheme_name = get_from_raw_row(raw_row, ["scheme_name", "Scheme Name", "scholarship_name"]) \
        or extract_field(formatted_text, ["Scheme Name", "Scholarship Name", "Scheme", "Scholarship Scheme"])
    category = get_from_raw_row(raw_row, ["category", "Category"]) \
        or extract_field(formatted_text, ["Category", "Categories"])

    metadata = {
        "source": source, "row": row, "file_type": file_type, "document_type": document_type,
        "scheme_name": scheme_name, "scheme_id": scheme_id, "category": category,
    }
    return {"text": formatted_text, "metadata": metadata}


def format_documents(documents):
    formatted = [format_document(d) for d in documents]
    return [d for d in formatted if d["text"].strip()]


def add_document_ids(documents):
    for i, document in enumerate(documents):
        document["metadata"]["document_id"] = f"doc_{i:06d}"
    return documents


# ---------------------------------------------------------------------------
# 5. Chunk
# ---------------------------------------------------------------------------
def split_into_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def create_chunks(text, chunk_size=350, chunk_overlap=50):
    if not text:
        return []
    if count_tokens(text) <= chunk_size:
        return [text.strip()]

    sentences = split_into_sentences(text)
    chunks, current_sentences, current_tokens = [], [], 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if sentence_tokens > chunk_size:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences, current_tokens = [], 0
            words, temp = sentence.split(), []
            for word in words:
                test = " ".join(temp + [word])
                if count_tokens(test) <= chunk_size:
                    temp.append(word)
                else:
                    if temp:
                        chunks.append(" ".join(temp))
                    temp = (temp[-10:] if temp else []) + [word]
            if temp:
                chunks.append(" ".join(temp))
            continue

        if current_tokens + sentence_tokens <= chunk_size:
            current_sentences.append(sentence)
            current_tokens += sentence_tokens
        else:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
            overlap_sentences, overlap_tokens = [], 0
            for previous in reversed(current_sentences):
                previous_tokens = count_tokens(previous)
                if overlap_tokens + previous_tokens <= chunk_overlap:
                    overlap_sentences.insert(0, previous)
                    overlap_tokens += previous_tokens
                else:
                    break
            current_sentences = overlap_sentences + [sentence]
            current_tokens = overlap_tokens + sentence_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))
    return [c.strip() for c in chunks if c.strip()]


def chunk_documents(documents, chunk_size=350, chunk_overlap=50):
    chunked_documents = []
    for document in documents:
        text = document["text"]
        metadata = document["metadata"].copy()
        chunks = create_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk_index, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = f"{metadata['document_id']}_chunk_{chunk_index}"
            chunk_metadata["chunk_index"] = chunk_index
            chunk_metadata["total_chunks"] = len(chunks)
            chunk_metadata["token_count"] = count_tokens(chunk)
            chunked_documents.append({"text": chunk, "metadata": chunk_metadata})
    return chunked_documents


# ---------------------------------------------------------------------------
# 6. Build the pipeline (runs once at import time)
# ---------------------------------------------------------------------------
print("Loading and processing documents...")
raw_documents = load_csvs(DATA_DIR) 
cleaned_documents = clean_documents(raw_documents)
formatted_documents = add_document_ids(format_documents(cleaned_documents))
chunked_documents = chunk_documents(formatted_documents, chunk_size=350, chunk_overlap=50)
print(f"Indexed {len(chunked_documents)} chunks from {len(formatted_documents)} documents")

print("Loading embedding model...")
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
embedding_dimension = embedding_model.get_embedding_dimension()

texts = [doc["text"] for doc in chunked_documents]
embeddings = embedding_model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
for document, embedding in zip(chunked_documents, embeddings):
    document["embedding"] = embedding.tolist()

print("Building BM25 index...")
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

bm25_corpus_tokens = [tokenize(doc["text"]) for doc in chunked_documents]
bm25_index = BM25Okapi(bm25_corpus_tokens)

print("Building Qdrant collection...")
client = QdrantClient(path=QDRANT_PATH)
atexit.register(lambda: client.close())
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE),
)
points = []
for i, (document, embedding) in enumerate(zip(chunked_documents, embeddings)):
    payload = {"text": document["text"], **document["metadata"]}
    points.append(PointStruct(id=i, vector=embedding.tolist(), payload=payload))
for start in range(0, len(points), 100):
    client.upsert(collection_name=COLLECTION_NAME, points=points[start:start + 100])

print("Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ---------------------------------------------------------------------------
# 7. Retrieval
# ---------------------------------------------------------------------------
def dense_search(query, top_k=30):
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    results = client.query_points(collection_name=COLLECTION_NAME, query=query_embedding.tolist(), limit=top_k).points
    retrieved = []
    for result in results:
        payload = result.payload
        retrieved.append({
            "text": payload.get("text", ""),
            "vector_score": float(result.score),
            "metadata": {
                "source": payload.get("source"), "row": payload.get("row"),
                "document_type": payload.get("document_type"), "scheme_name": payload.get("scheme_name"),
                "scheme_id": payload.get("scheme_id"), "category": payload.get("category"),
                "document_id": payload.get("document_id"), "chunk_id": payload.get("chunk_id"),
            },
        })
    return retrieved


def keyword_search(query, top_k=30):
    scores = bm25_index.get_scores(tokenize(query))
    ranked_idx = np.argsort(scores)[::-1][:top_k]
    retrieved = []
    for idx in ranked_idx:
        if scores[idx] <= 0:
            continue
        document = chunked_documents[idx]
        retrieved.append({"text": document["text"], "bm25_score": float(scores[idx]), "metadata": document["metadata"]})
    return retrieved


def scheme_filtered_search(query, scheme_ids, top_k=15):
    if not scheme_ids:
        return []
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    scheme_filter = Filter(must=[FieldCondition(key="scheme_id", match=MatchAny(any=scheme_ids))])
    results = client.query_points(
        collection_name=COLLECTION_NAME, query=query_embedding.tolist(), query_filter=scheme_filter, limit=top_k,
    ).points
    retrieved = []
    for result in results:
        payload = result.payload
        retrieved.append({
            "text": payload.get("text", ""), "vector_score": float(result.score), "bm25_score": 0.0,
            "metadata": {
                "source": payload.get("source"), "row": payload.get("row"),
                "document_type": payload.get("document_type"), "scheme_name": payload.get("scheme_name"),
                "scheme_id": payload.get("scheme_id"), "category": payload.get("category"),
                "document_id": payload.get("document_id"), "chunk_id": payload.get("chunk_id"),
            },
        })
    return retrieved


def hybrid_retrieve(query, vector_k=30, bm25_k=30, scheme_k=15):
    dense_results = dense_search(query, top_k=vector_k)
    keyword_results = keyword_search(query, top_k=bm25_k)
    detected_schemes = detect_scheme(query)
    scheme_results = scheme_filtered_search(query, detected_schemes, top_k=scheme_k)

    merged = {}
    for r in dense_results:
        merged[r["metadata"]["chunk_id"]] = {**r, "bm25_score": 0.0}
    for r in keyword_results:
        chunk_id = r["metadata"]["chunk_id"]
        if chunk_id in merged:
            merged[chunk_id]["bm25_score"] = r["bm25_score"]
        else:
            merged[chunk_id] = {**r, "vector_score": 0.0}
    for r in scheme_results:
        chunk_id = r["metadata"]["chunk_id"]
        if chunk_id not in merged:
            merged[chunk_id] = r
    return list(merged.values())


def rerank_documents(query, documents):
    if not documents:
        return []
    pairs = [[query, doc["text"]] for doc in documents]
    scores = reranker.predict(pairs)
    for document, score in zip(documents, scores):
        document["rerank_score"] = float(score)
    return documents


# ---------------------------------------------------------------------------
# 8. Query understanding
# ---------------------------------------------------------------------------
SCHEME_KEYWORDS = {
    "SCH01": ["post matric", "post-matric", "post matric scholarship", "pms", "college scholarship", "degree scholarship"],
    "SCH07": ["pre matric classes ix", "pre matric classes x"],
    "SCH08": ["pre matric", "pre-matric", "pre matric scholarship"],
    "SCH03": ["overseas", "foreign university", "abroad", "minorities overseas"],
    "SCH02": ["ambedkar overseas"],
}

TOPIC_KEYWORDS = {
    "income": ["income", "income limit", "family income", "annual income"],
    "attendance": ["attendance", "biometric", "75%"],
    "rejection": ["reject", "rejected", "rejection", "reason for rejection", "denied", "ineligible"],
    "quota": ["quota", "management quota", "category-b", "category b", "spot admission", "convener quota"],
    "documents": ["document", "certificate", "bonafide", "income certificate", "caste certificate"],
    "renewal": ["renewal", "renew", "previous year"],
    "bank": ["bank account", "passbook", "payment", "mtf"],
}


def detect_scheme(query):
    query_lower = query.lower()
    return [sid for sid, kws in SCHEME_KEYWORDS.items() if any(k in query_lower for k in kws)]


def detect_topics(query):
    query_lower = query.lower()
    return [t for t, kws in TOPIC_KEYWORDS.items() if any(k in query_lower for k in kws)]


def classify_query(query):
    schemes = detect_scheme(query)
    topics = detect_topics(query)
    return {"schemes": schemes, "topics": topics, "scheme_specified": len(schemes) > 0}


# ---------------------------------------------------------------------------
# 9. Final ranking
# ---------------------------------------------------------------------------
def keyword_overlap_score(query, text):
    query_words = set(re.findall(r"\b[a-zA-Z0-9]+\b", query.lower()))
    text_words = set(re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()))
    if not query_words:
        return 0.0
    return len(query_words & text_words) / len(query_words)


def scheme_match_score(query, metadata):
    detected_schemes = detect_scheme(query)
    if not detected_schemes:
        return 0.0
    return 1.0 if metadata.get("scheme_id") in detected_schemes else 0.0


def topic_match_score(query, text):
    topics = detect_topics(query)
    if not topics:
        return 0.0
    text_lower = text.lower()
    matches = sum(1 for topic in topics if any(k in text_lower for k in TOPIC_KEYWORDS[topic]))
    return min(matches / len(topics), 1.0)


def final_rank(query, documents, top_k=5):
    ranked = []
    for document in documents:
        vector_score = document.get("vector_score", 0.0)
        bm25_score = document.get("bm25_score", 0.0)
        rerank_score = document.get("rerank_score", 0.0)

        rerank_normalized = 1 / (1 + math.exp(-rerank_score))
        bm25_normalized = min(bm25_score / 10.0, 1.0)

        scheme_score = scheme_match_score(query, document["metadata"])
        topic_score = topic_match_score(query, document["text"])
        keyword_score = keyword_overlap_score(query, document["text"])

        final_score = (
            0.20 * vector_score + 0.15 * bm25_normalized + 0.25 * rerank_normalized
            + 0.15 * topic_score + 0.15 * scheme_score + 0.10 * keyword_score
        )

        document["scheme_score"] = scheme_score
        document["topic_score"] = topic_score
        document["keyword_score"] = keyword_score
        document["final_score"] = final_score
        ranked.append(document)

    ranked.sort(key=lambda x: x["final_score"], reverse=True)

    detected_schemes = detect_scheme(query)
    detected_topics = detect_topics(query)
    if detected_schemes and detected_topics:
        exact_match_ids = {d["metadata"]["chunk_id"] for d in ranked if d["scheme_score"] == 1.0 and d["topic_score"] == 1.0}
        if exact_match_ids:
            exact_matches = [d for d in ranked if d["metadata"]["chunk_id"] in exact_match_ids]
            others = [d for d in ranked if d["metadata"]["chunk_id"] not in exact_match_ids]
            selected = (exact_matches + others)[:top_k]
            selected.sort(key=lambda x: x["final_score"], reverse=True)
            return selected

    return ranked[:top_k]


def build_context(results):
    context_parts = []
    for i, result in enumerate(results):
        metadata = result["metadata"]
        header = f"SOURCE {i + 1}\nSource File: {metadata.get('source', 'Unknown')}\nDocument Type: {metadata.get('document_type', 'Unknown')}\n"
        if metadata.get("scheme_id"):
            header += f"Scheme ID: {metadata['scheme_id']}\n"
        if metadata.get("category"):
            header += f"Category: {metadata['category']}\n"
        context_parts.append(header + "\nContent:\n" + result["text"])
    return "\n\n" + "\n\n".join(context_parts)


# ---------------------------------------------------------------------------
# 10. LLM (Groq)
# ---------------------------------------------------------------------------
load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("Groq API key could not be loaded")
client_llm = Groq(api_key=groq_api_key)

SYSTEM_PROMPT = """You are a Telangana Scholarship Information Assistant.

Your job is to answer questions ONLY using the retrieved official Telangana scholarship
context provided to you.

STRICT RULES:
1. Never invent information.
2. Never use outside knowledge.
3. Do not combine rules from different scholarship schemes unless the user explicitly asks for a comparison.
4. If a scheme is explicitly mentioned by the user, prioritize information belonging to that scheme.
5. If the user does not specify a scheme and the retrieved context contains different values for different schemes, clearly explain that the answer depends on the scheme.
6. Do not present a scheme-specific value as a universal value.
7. Preserve exact numbers, percentages, income limits and conditions from the retrieved context.
8. If the retrieved context is insufficient, say: "I could not find sufficient information in the available official sources."
9. Do not create citations, rules, scheme IDs or source files that are not present in the context.
10. Keep answers concise and easy to understand.
11. When useful, mention the source file the information came from.
12. If two sources contain different values, do NOT silently choose one. State that the sources differ and identify the relevant source/scheme for each value.
"""


def generate_answer(query, context, conversation_context="", max_retries=4):
    user_prompt = f"""{conversation_context}CURRENT USER QUESTION:
{query}

RETRIEVED CONTEXT (this is your ONLY source of facts — never the conversation above):
{context}

Answer the CURRENT question using only the retrieved context.
"""
    for attempt in range(max_retries):
        try:
            response = client_llm.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                time.sleep(2 ** attempt)
            else:
                raise
    raise RuntimeError("Groq API unavailable after retries.")

GREETING_PATTERNS = {"hi", "hello", "hey", "okay", "ok", "thanks", "thank you", "bye", "goodbye"}

def is_chitchat(query):
    """Catch greetings/acknowledgments and vague one-word inputs that don't need retrieval."""
    stripped = query.strip().lower().rstrip("!.,?")
    if stripped in GREETING_PATTERNS:
        return True
    # very short, no real content word (e.g. "how", "why", "ok then")
    if len(stripped.split()) <= 2 and not detect_scheme(query) and not detect_topics(query):
        return True
    return False


def handle_chitchat(query):
    response = client_llm.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": (
                "You are a friendly assistant for a Telangana scholarship chatbot. "
                "The user sent a greeting, acknowledgment, or vague message with no "
                "specific scholarship question. Respond briefly and naturally, and "
                "invite them to ask about eligibility, income limits, documents, "
                "attendance rules, or rejection reasons. Do not invent scholarship facts."
            )},
            {"role": "user", "content": query},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content
# ---------------------------------------------------------------------------
# 11. Full pipeline + memory helpers (what app.py imports)
# ---------------------------------------------------------------------------
def ask_scholarship_assistant(query, vector_k=30, bm25_k=30, final_k=5):
    if is_chitchat(query):
        answer = handle_chitchat(query)
        return {"query": query, "query_info": {"schemes": [], "topics": [], "scheme_specified": False},
                "answer": answer, "sources": [], "context": ""}

    query_info = classify_query(query)
    candidates = hybrid_retrieve(query, vector_k=vector_k, bm25_k=bm25_k)
    candidates = rerank_documents(query, candidates)
    final_results = final_rank(query, candidates, top_k=final_k)
    context = build_context(final_results)
    answer = generate_answer(query, context)
    return {"query": query, "query_info": query_info, "answer": answer, "sources": final_results, "context": context}

print("rag_pipeline.py ready.")