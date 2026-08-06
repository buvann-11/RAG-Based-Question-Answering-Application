from __future__ import annotations

import re
from dataclasses import dataclass

from .config import MAX_CITATIONS
from .llm import generate_grounded_answer
from .retrieval import RetrievedChunk, informative_tokens, retrieve_chunks


SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\s+(?=•)")
ROMAN_ITEM_PATTERN = re.compile(r"\b(i|ii|iii|iv|v|vi|vii|viii|ix|x)\.\s*", re.IGNORECASE)
STACK_PATTERN = re.compile(
    r"(frontend|backend|document processing|vector store|llm integration|context management|response generation|query processing|streamlit|langchain|chromadb|pypdfloader|firebase|gemini|pinecone)",
    re.IGNORECASE,
)
CATEGORY_LINE_PATTERN = re.compile(
    r"^(frontend|backend|document processing|vector store integration|llm integration|context management|response generation|query processing)\b",
    re.IGNORECASE,
)


@dataclass
class AnswerResult:
    answer: str
    citations: list[dict]
    retrievals: list[RetrievedChunk]
    grounded: bool
    answer_engine: str


def _is_stack_question(question: str) -> bool:
    lowered = question.lower()
    return any(
        phrase in lowered
        for phrase in [
            "tech stack",
            "technology stack",
            "tools",
            "framework",
            "libraries",
            "frontend",
            "backend",
            "built with",
        ]
    )


def _normalize_sentence(sentence: str) -> str:
    sentence = re.sub(r"\s+", " ", sentence).strip(" -:")
    sentence = sentence.replace("• ", "")
    sentence = re.sub(
        r"^A Tools Methods The paper uses following tools as well as methods as\s*",
        "",
        sentence,
        flags=re.IGNORECASE,
    )
    return sentence.strip()


def _extract_candidates(chunk: RetrievedChunk) -> list[str]:
    candidates: list[str] = []
    for sentence in SENTENCE_PATTERN.split(chunk.content):
        cleaned = _normalize_sentence(sentence)
        if cleaned:
            candidates.append(cleaned)
    return candidates


def _chunk_support_score(question: str, chunk: RetrievedChunk) -> float:
    query_terms = set(informative_tokens(question))
    chunk_terms = set(informative_tokens(chunk.content))
    overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1)
    score = chunk.combined_score + overlap
    if "definition" in question.lower() and re.search(r"\bdefinition|defined|means\b", chunk.content, re.IGNORECASE):
        score += 0.45
    return score


def _sentence_score(question: str, sentence: str, chunk_score: float) -> float:
    query_terms = set(informative_tokens(question))
    sentence_terms = set(informative_tokens(sentence))
    if not sentence_terms:
        return 0.0

    overlap = len(query_terms & sentence_terms)
    coverage = overlap / max(len(query_terms), 1)
    definition_bonus = 0.45 if "definition" in question.lower() and re.search(r"\bdefinition|defined|means\b", sentence, re.IGNORECASE) else 0.0
    stack_bonus = 0.5 if _is_stack_question(question) and STACK_PATTERN.search(sentence) else 0.0
    category_bonus = 1.0 if _is_stack_question(question) and CATEGORY_LINE_PATTERN.search(sentence) else 0.0
    named_tool_bonus = 0.35 if _is_stack_question(question) and re.search(
        r"\b(streamlit|langchain|chromadb|pypdfloader|firebase|gemini|pinecone)\b",
        sentence,
        re.IGNORECASE,
    ) else 0.0
    return (coverage * 3.0) + chunk_score + definition_bonus + stack_bonus + category_bonus + named_tool_bonus


def _deduplicate_sentences(candidates: list[tuple[float, str, RetrievedChunk]]) -> list[tuple[float, str, RetrievedChunk]]:
    unique: list[tuple[float, str, RetrievedChunk]] = []
    seen_keys: set[str] = set()
    for score, sentence, chunk in candidates:
        key = " ".join(sorted(set(informative_tokens(sentence))))[:200]
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append((score, sentence, chunk))
    return unique


def _fallback_answer(question: str, citations: list[dict]) -> str:
    if _is_stack_question(question):
        lines = ["According to the document excerpts:"]
        for citation in citations:
            lines.append(f"- {citation['excerpt']} [{citation['index']}]")
        return "\n".join(lines)
    return " ".join(f"{citation['excerpt']} [{citation['index']}]" for citation in citations)


def _extract_roman_items(text: str) -> list[str]:
    subject_match = re.search(r"subject to", text, re.IGNORECASE)
    if subject_match:
        text = text[subject_match.end():]
    matches = list(ROMAN_ITEM_PATTERN.finditer(text))
    items: list[str] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        item = _normalize_sentence(text[start:end])
        if len(item) >= 6 and len(item) <= 220 and "if you are advised hospitalization" not in item.lower():
            items.append(item.rstrip("."))
    return items


def _extractive_grounded_answer(question: str, citations: list[dict]) -> str | None:
    lowered = question.lower()
    if "pre-hospitalization" in lowered or "pre hospitalization" in lowered:
        for citation in citations:
            excerpt = citation["excerpt"]
            match = re.search(
                r"(pre-hospitalization.*?hospitalization claim under inpatient hospitalization treatment\.?)",
                excerpt,
                re.IGNORECASE | re.DOTALL,
            )
            if match:
                answer = _normalize_sentence(match.group(1))
                return f"{answer} [{citation['index']}]"

    if "post-hospitalization" in lowered or "post hospitalization" in lowered:
        for citation in citations:
            excerpt = citation["excerpt"]
            match = re.search(
                r"(post-hospitalization.*?hospitalization claim under inpatient hospitalization treatment\.?)",
                excerpt,
                re.IGNORECASE | re.DOTALL,
            )
            if match:
                answer = _normalize_sentence(match.group(1))
                return f"{answer} [{citation['index']}]"

    if "exclusion" in lowered or "excluded" in lowered:
        for citation in citations:
            excerpt = citation["excerpt"]
            if any(term in excerpt.lower() for term in ["exclusion", "excluded", "not covered", "shall be excluded"]):
                short_excerpt = _normalize_sentence(excerpt[:420])
                return (
                    "The policy contains multiple exclusions under the exclusions section. "
                    f"One retrieved exclusion clause is: {short_excerpt} [{citation['index']}] "
                    "For a more precise answer, ask about a specific exclusion item."
                )

    if any(phrase in lowered for phrase in ["expenses", "covered", "benefits", "what is covered"]):
        collected_items: list[tuple[int, str]] = []
        for citation in citations:
            items = _extract_roman_items(citation["excerpt"])
            if items:
                for item in items:
                    collected_items.append((citation["index"], item))
        if collected_items:
            lines = ["According to the uploaded document, the covered items include:"]
            seen_items: set[str] = set()
            for citation_index, item in collected_items:
                key = item.lower()
                if key in seen_items:
                    continue
                seen_items.add(key)
                lines.append(f"- {item} [{citation_index}]")
            return "\n".join(lines)

    if "definition" in lowered:
        match = re.search(r"definition of ([a-z][a-z\s\-]+?)(?: in the policy|$)", lowered)
        definition_term = match.group(1).strip() if match else ""
        definition_term = re.sub(r"^(a|an|the)\s+", "", definition_term)
        for citation in citations:
            excerpt = citation["excerpt"]
            if definition_term:
                pattern = re.compile(
                    rf"(?:\d+\.\s*)?{re.escape(definition_term)}\s*[:-]\s*(.+?)(?=(?:\d+\.\s*[A-Z][A-Za-z\s/()'-]*[:-])|$)",
                    re.IGNORECASE | re.DOTALL,
                )
                term_match = pattern.search(excerpt)
                if term_match:
                    answer = _normalize_sentence(f"{definition_term.title()}: {term_match.group(1)}")
                    return f"{answer} [{citation['index']}]"
            if re.search(r"\b(means|shall mean|is defined as|shall be construed as)\b", excerpt, re.IGNORECASE):
                return f"{excerpt} [{citation['index']}]"

    return None


def _best_sentence_for_chunk(question: str, chunk: RetrievedChunk) -> str:
    candidates = _extract_candidates(chunk)
    if not candidates:
        return _normalize_sentence(chunk.content[:420])
    best = max(
        candidates,
        key=lambda sentence: _sentence_score(question, sentence, chunk.combined_score),
    )
    return best


def _build_citations(question: str, retrievals: list[RetrievedChunk]) -> list[dict]:
    citations: list[dict] = []
    used_chunk_ids: set[int] = set()
    citation_limit = MAX_CITATIONS if _is_stack_question(question) else 2

    for chunk in retrievals:
        if chunk.chunk_id in used_chunk_ids:
            continue
        used_chunk_ids.add(chunk.chunk_id)
        excerpt = _best_sentence_for_chunk(question, chunk)
        if not _is_stack_question(question):
            excerpt = _normalize_sentence(chunk.content[:1200])
        citations.append(
            {
                "index": len(citations) + 1,
                "filename": chunk.filename,
                "page_label": chunk.page_label,
                "excerpt": excerpt,
                "combined_score": round(chunk.combined_score, 3),
                "lexical_score": round(chunk.lexical_score, 3),
                "semantic_score": round(chunk.semantic_score, 3),
            }
        )
        if len(citations) >= citation_limit:
            break
    return citations


def answer_question(user_id: int, question: str) -> AnswerResult:
    retrievals = retrieve_chunks(user_id, question, top_k=6)
    if not retrievals:
        return AnswerResult(
            answer="No indexed document content is available for this account yet.",
            citations=[],
            retrievals=[],
            grounded=False,
            answer_engine="No context",
        )

    citations = _build_citations(question, retrievals)
    if not citations:
        return AnswerResult(
            answer="I couldn't find the answer in the retrieved document excerpts.",
            citations=[],
            retrievals=retrievals,
            grounded=False,
            answer_engine="Retrieval only",
        )

    extractive_answer = _extractive_grounded_answer(question, citations)
    if extractive_answer is not None:
        return AnswerResult(
            answer=extractive_answer,
            citations=citations,
            retrievals=retrievals,
            grounded=True,
            answer_engine="Extractive grounded",
        )

    llm_answer = generate_grounded_answer(question, citations)
    if llm_answer is not None:
        if llm_answer.answer.strip().lower().startswith("i couldn't find the answer"):
            return AnswerResult(
                answer="I couldn't find the answer in the uploaded document.",
                citations=[],
                retrievals=[],
                grounded=False,
                answer_engine=f"Ollama {llm_answer.model_name}",
            )
        return AnswerResult(
            answer=llm_answer.answer,
            citations=citations,
            retrievals=retrievals,
            grounded=True,
            answer_engine=f"Ollama {llm_answer.model_name}",
        )

    if _is_stack_question(question):
        return AnswerResult(
            answer=_fallback_answer(question, citations),
            citations=citations,
            retrievals=retrievals,
            grounded=True,
            answer_engine="Extractive fallback",
        )

    return AnswerResult(
        answer="I couldn't find the answer in the uploaded document.",
        citations=[],
        retrievals=[],
        grounded=False,
        answer_engine="No grounded answer",
    )
