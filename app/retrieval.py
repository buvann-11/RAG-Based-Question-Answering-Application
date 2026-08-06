from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from .db import get_connection
from .embeddings import deserialize_vector, get_embedder


TOKEN_PATTERN = re.compile(r"\b\w+\b")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "into", "is", "it", "of", "on", "or", "the", "this", "to", "what",
    "which", "who", "why", "with", "tell", "me", "about", "does", "do", "used",
    "under", "policy", "document", "uploaded",
}
STACK_HINTS = {
    "stack", "tech", "technology", "tools", "method", "methods", "framework",
    "frameworks", "frontend", "backend", "database", "vector", "embedding",
    "model", "deployment", "libraries", "library", "authentication",
}
POLICY_INTENTS = {
    "definition": {
        "triggers": ["definition", "define", "defined", "meaning", "means", "construed"],
        "query_hints": "standard definitions specific definitions means defined shall mean shall be construed",
        "chunk_terms": ["definition", "definitions", "defined", "means", "shall mean", "shall be construed"],
        "heading_terms": ["standard definitions", "specific definitions"],
    },
    "coverage": {
        "triggers": ["covered", "coverage", "benefits", "expenses", "claim"],
        "query_hints": "benefits covered under the policy scope of cover reasonable and customary expenses",
        "chunk_terms": ["covered", "coverage", "benefits", "expenses", "reasonable", "customary"],
        "heading_terms": ["benefits covered under the policy", "scope of cover"],
    },
    "inpatient": {
        "triggers": ["in-patient", "inpatient", "hospitalization", "icu", "room rent", "nursing"],
        "query_hints": (
            "in-patient hospitalization treatment room rent boarding icu nursing surgeon "
            "anesthesia oxygen operation theatre medicines drugs diagnostic tests"
        ),
        "chunk_terms": [
            "in-patient hospitalization treatment",
            "hospitalization treatment",
            "room rent",
            "boarding",
            "icu",
            "nursing",
            "surgeon",
            "anesthesia",
            "medicines",
            "drugs",
        ],
        "heading_terms": ["in-patient hospitalization treatment", "in-patient benefits"],
    },
    "pre_hospitalization": {
        "triggers": ["pre-hospitalization", "pre hospitalization"],
        "query_hints": "pre-hospitalization immediately before hospitalized same illness injury",
        "chunk_terms": ["pre-hospitalization", "pre hospitalization"],
        "heading_terms": ["pre-hospitalization"],
    },
    "post_hospitalization": {
        "triggers": ["post-hospitalization", "post hospitalization"],
        "query_hints": "post-hospitalization immediately after discharge",
        "chunk_terms": ["post-hospitalization", "post hospitalization"],
        "heading_terms": ["post-hospitalization"],
    },
    "exclusion": {
        "triggers": ["exclusion", "excluded", "not covered", "shall not", "not payable"],
        "query_hints": "exclusions standard exclusions not covered shall not payable",
        "chunk_terms": ["exclusion", "excluded", "shall not", "not covered", "not payable"],
        "heading_terms": ["exclusions", "standard exclusions"],
    },
    "waiting_period": {
        "triggers": ["waiting period", "wait period"],
        "query_hints": "waiting period applicable waiting",
        "chunk_terms": ["waiting period", "applicable waiting"],
        "heading_terms": ["waiting period"],
    },
    "sum_insured": {
        "triggers": ["sum insured", "si"],
        "query_hints": "sum insured maximum liability",
        "chunk_terms": ["sum insured", "maximum liability"],
        "heading_terms": ["sum insured"],
    },
}


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def informative_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS and len(token) > 2]


def _matched_intents(query: str) -> list[str]:
    lowered = query.lower()
    matched: list[str] = []
    for intent_name, intent in POLICY_INTENTS.items():
        if any(trigger in lowered for trigger in intent["triggers"]):
            matched.append(intent_name)
    return matched


def expand_query(query: str) -> str:
    lowered = query.lower()
    hints: list[str] = []
    if "tech stack" in lowered or ("tech" in lowered and "stack" in lowered):
        hints.extend(
            [
                "frontend backend framework tools methods libraries database vector store embedding model deployment",
                "streamlit langchain chromadb pypdfloader firebase gemini",
            ]
        )
    if "how" in lowered and "work" in lowered:
        hints.append("pipeline retrieval embeddings chunks citations answer generation")
    for intent_name in _matched_intents(query):
        hints.append(POLICY_INTENTS[intent_name]["query_hints"])
    return f"{query} {' '.join(hints)}".strip()


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    filename: str
    page_label: str
    content: str
    lexical_score: float
    semantic_score: float
    combined_score: float


class BM25Index:
    def __init__(self, documents: Iterable[str]) -> None:
        tokenized_docs = [tokenize(document) for document in documents]
        self.documents = tokenized_docs
        self.doc_count = len(tokenized_docs)
        self.avgdl = (
            sum(len(document) for document in tokenized_docs) / self.doc_count
            if self.doc_count
            else 0.0
        )
        self.doc_freq = defaultdict(int)
        for document in tokenized_docs:
            for token in set(document):
                self.doc_freq[token] += 1

    def score(self, query: str) -> list[float]:
        query_tokens = tokenize(query)
        scores = []
        for document in self.documents:
            frequencies = Counter(document)
            score = 0.0
            length = len(document) or 1
            for token in query_tokens:
                if token not in frequencies:
                    continue
                idf = math.log(
                    1 + (self.doc_count - self.doc_freq[token] + 0.5)
                    / (self.doc_freq[token] + 0.5)
                )
                tf = frequencies[token]
                numerator = tf * 2.2
                denominator = tf + 1.2 * (1 - 0.75 + 0.75 * length / max(self.avgdl, 1))
                score += idf * (numerator / denominator)
            scores.append(score)
        return scores


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        return [1.0 if maximum > 0 else 0.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def _coverage_score(query: str, text: str) -> float:
    query_tokens = set(informative_tokens(query))
    text_tokens = set(informative_tokens(text))
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens) / len(query_tokens)
    stack_overlap = len(STACK_HINTS & text_tokens) / max(len(STACK_HINTS), 1)
    return overlap + (0.45 * stack_overlap)


def _policy_section_score(query: str, text: str) -> float:
    lowered_query = query.lower()
    lowered_text = text.lower()
    score = 0.0

    for intent_name in _matched_intents(query):
        intent = POLICY_INTENTS[intent_name]
        if any(term in lowered_text for term in intent["heading_terms"]):
            score += 1.2
        term_hits = sum(1 for term in intent["chunk_terms"] if term in lowered_text)
        score += min(term_hits * 0.35, 1.4)

    if "definition" in lowered_query and "standard definitions" in lowered_text:
        score += 0.9
    if "definition" in lowered_query and re.search(r"\b(shall mean|means|defined as|shall be construed)\b", lowered_text):
        score += 0.8

    if any(term in lowered_query for term in ["in-patient", "inpatient", "hospitalization treatment"]):
        if "in-patient hospitalization treatment" in lowered_text:
            score += 1.6
        if "section c" in lowered_text and "benefits covered under the policy" in lowered_text:
            score += 0.8

    if "exclusion" in lowered_query and "exclusions" in lowered_text:
        score += 1.4

    return score


def _direct_clause_bonus(query: str, text: str) -> float:
    lowered_query = " ".join(query.lower().split())
    lowered_text = " ".join(text.lower().split())
    bonus = 0.0

    if "pre-hospitalization" in lowered_query or "pre hospitalization" in lowered_query:
        if "pre-hospitalization" in lowered_text or "pre hospitalization" in lowered_text:
            bonus += 0.7

    if "post-hospitalization" in lowered_query or "post hospitalization" in lowered_query:
        if "post-hospitalization" in lowered_text or "post hospitalization" in lowered_text:
            bonus += 0.7

    if "in-patient hospitalization treatment" in lowered_query and "in-patient hospitalization treatment" in lowered_text:
        bonus += 0.45

    if "definition of" in lowered_query and "standard definitions" in lowered_text:
        bonus += 0.45

    if lowered_query in lowered_text and len(lowered_query) > 18:
        bonus += 0.35

    return bonus


def retrieve_chunks(user_id: int, query: str, top_k: int = 5) -> list[RetrievedChunk]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT chunks.id, chunks.document_id, chunks.content, chunks.page_label,
                   chunks.embedding, documents.filename
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            WHERE documents.user_id = ?
            """,
            (user_id,),
        ).fetchall()

    if not rows:
        return []

    contents = [row["content"] for row in rows]
    bm25 = BM25Index(contents)
    expanded_query = expand_query(query)
    lexical_scores = bm25.score(expanded_query)

    embedder = get_embedder()
    query_vector = embedder.encode([query])[0]
    semantic_scores = []
    coverage_scores = []
    section_scores = []
    direct_clause_bonuses = []
    for row in rows:
        vector = deserialize_vector(row["embedding"])
        score = float(np.dot(query_vector, vector))
        semantic_scores.append(score)
        coverage_scores.append(_coverage_score(query, row["content"]))
        section_scores.append(_policy_section_score(query, row["content"]))
        direct_clause_bonuses.append(_direct_clause_bonus(query, row["content"]))

    lexical_norm = _normalize(lexical_scores)
    semantic_norm = _normalize(semantic_scores)
    coverage_norm = _normalize(coverage_scores)
    section_norm = _normalize(section_scores)

    results = []
    for index, row in enumerate(rows):
        combined = (
            0.34 * semantic_norm[index]
            + 0.28 * lexical_norm[index]
            + 0.18 * coverage_norm[index]
            + 0.20 * section_norm[index]
            + direct_clause_bonuses[index]
        )
        results.append(
            RetrievedChunk(
                chunk_id=row["id"],
                document_id=row["document_id"],
                filename=row["filename"],
                page_label=row["page_label"] or "?",
                content=row["content"],
                lexical_score=lexical_scores[index],
                semantic_score=semantic_scores[index],
                combined_score=combined,
            )
        )
    return sorted(results, key=lambda item: item.combined_score, reverse=True)[:top_k]
