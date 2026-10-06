from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable
from urllib import error, request


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_CANDIDATES = ["phi3:mini", "llama3.2:1b"]
UNAVAILABLE_MODELS: set[str] = set()


@dataclass
class LLMAnswer:
    answer: str
    model_name: str


def _clean_response(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _build_prompt(question: str, citations: Iterable[dict]) -> str:
    citation_blocks = []
    for citation in citations:
        citation_blocks.append(
            (
                f"[{citation['index']}] {citation['filename']} page {citation['page_label']}\n"
                f"{citation['excerpt']}"
            )
        )

    excerpts = "\n\n".join(citation_blocks)
    return (
        "You are an offline document question answering assistant.\n"
        "Answer only from the provided excerpts.\n"
        "If the excerpts do not contain the answer, say: "
        "\"I couldn't find the answer in the retrieved document excerpts.\"\n"
        "If the excerpts only partially answer the question, explicitly say that the answer is partial.\n"
        "Do not infer, guess, or fill in missing definitions.\n"
        "If the excerpts only cross-reference another section or definition, say that clearly instead of inferring the missing text.\n"
        "Keep the answer concise, accurate, and easy to read.\n"
        "Every factual sentence must include one or more citation markers like [1] or [2].\n"
        "Do not invent details.\n\n"
        f"Question:\n{question}\n\n"
        f"Excerpts:\n{excerpts}\n\n"
        "Answer:"
    )


def _call_ollama(model_name: str, prompt: str) -> str:
    payload = json.dumps(
        {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 180,
                "num_ctx": 2048,
            },
        }
    ).encode("utf-8")

    http_request = request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(http_request, timeout=90) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body.get("response", "")).strip()


def generate_grounded_answer(question: str, citations: list[dict]) -> LLMAnswer | None:
    if not citations:
        return None

    prompt = _build_prompt(question, citations)
    for model_name in MODEL_CANDIDATES:
        if model_name in UNAVAILABLE_MODELS:
            continue
        try:
            answer = _clean_response(_call_ollama(model_name, prompt))
            if not answer:
                continue
            return LLMAnswer(answer=answer, model_name=model_name)
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            if "requires more system memory" in details.lower():
                UNAVAILABLE_MODELS.add(model_name)
                continue
        except Exception:
            continue

    return None
