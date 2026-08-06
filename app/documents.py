from __future__ import annotations

import mimetypes
import re
import shutil
from pathlib import Path
from typing import Iterable, List

from docx import Document as DocxDocument
from pypdf import PdfReader

from .config import CHUNK_OVERLAP, CHUNK_SIZE, UPLOAD_DIR
from .db import connection_scope, get_connection
from .embeddings import get_embedder, serialize_vector


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")


def save_upload(user_id: int, upload_name: str, source_stream) -> Path:
    target_dir = UPLOAD_DIR / str(user_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(upload_name).name
    target_path = target_dir / safe_name
    with target_path.open("wb") as handle:
        shutil.copyfileobj(source_stream, handle)
    return target_path


def extract_document_segments(path: Path) -> List[dict]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [{"page_label": "1", "text": text}]

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [
            {
                "page_label": str(index + 1),
                "text": page.extract_text() or "",
            }
            for index, page in enumerate(reader.pages)
        ]

    doc = DocxDocument(str(path))
    text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    return [{"page_label": "1", "text": text}]


def normalize_extracted_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = text.replace("•", "\n• ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n|•)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text_units(text: str) -> list[str]:
    normalized = normalize_extracted_text(text)
    if not normalized:
        return []

    units: list[str] = []
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n{2,}", normalized)
        if paragraph.strip()
    ]
    for paragraph in paragraphs:
        if paragraph.startswith("•"):
            units.extend(
                item.strip()
                for item in re.split(r"\n(?=•)", paragraph)
                if item.strip()
            )
            continue
        units.extend(
            sentence.strip()
            for sentence in SENTENCE_BOUNDARY_PATTERN.split(paragraph)
            if sentence.strip()
        )
    return units


def chunk_text(text: str) -> Iterable[str]:
    units = split_text_units(text)
    if not units:
        return []

    chunks: list[str] = []
    current_units: list[str] = []
    current_length = 0

    for unit in units:
        clean_unit = re.sub(r"\s+", " ", unit).strip()
        if not clean_unit:
            continue

        projected_length = current_length + len(clean_unit) + (1 if current_units else 0)
        if current_units and projected_length > CHUNK_SIZE:
            chunks.append(" ".join(current_units))
            overlap_units: list[str] = []
            overlap_length = 0
            for previous in reversed(current_units):
                overlap_units.insert(0, previous)
                overlap_length += len(previous) + 1
                if overlap_length >= CHUNK_OVERLAP:
                    break
            current_units = overlap_units
            current_length = sum(len(item) for item in current_units) + max(len(current_units) - 1, 0)

        current_units.append(clean_unit)
        current_length += len(clean_unit) + (1 if len(current_units) > 1 else 0)

    if current_units:
        chunks.append(" ".join(current_units))

    return chunks


def ingest_document(user_id: int, filename: str, stored_path: Path) -> int:
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    segments = extract_document_segments(stored_path)
    chunk_rows = []
    for segment in segments:
        for chunk in chunk_text(segment["text"]):
            chunk_rows.append(
                {
                    "content": chunk,
                    "page_label": segment["page_label"],
                }
            )

    embedder = get_embedder()
    embeddings = embedder.encode(row["content"] for row in chunk_rows)

    with connection_scope() as connection:
        cursor = connection.execute(
            """
            INSERT INTO documents(user_id, filename, source_path, mime_type)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, filename, str(stored_path), mime_type),
        )
        document_id = cursor.lastrowid
        for index, row in enumerate(chunk_rows):
            connection.execute(
                """
                INSERT INTO chunks(document_id, chunk_index, content, page_label, embedding)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    index,
                    row["content"],
                    row["page_label"],
                    serialize_vector(embeddings[index]),
                ),
            )
    return document_id


def list_documents(user_id: int) -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, filename, mime_type, created_at
            FROM documents
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_document(user_id: int, document_id: int) -> None:
    with connection_scope() as connection:
        row = connection.execute(
            """
            SELECT source_path
            FROM documents
            WHERE id = ? AND user_id = ?
            """,
            (document_id, user_id),
        ).fetchone()
        if not row:
            return

        connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM documents WHERE id = ? AND user_id = ?", (document_id, user_id))

    source_path = Path(row["source_path"])
    if source_path.exists():
        source_path.unlink(missing_ok=True)


def reindex_user_documents(user_id: int) -> int:
    with connection_scope() as connection:
        rows = connection.execute(
            """
            SELECT id, filename, source_path
            FROM documents
            WHERE user_id = ?
            ORDER BY id ASC
            """,
            (user_id,),
        ).fetchall()

        if not rows:
            return 0

        connection.execute(
            """
            DELETE FROM chunks
            WHERE document_id IN (
                SELECT id FROM documents WHERE user_id = ?
            )
            """,
            (user_id,),
        )

        embedder = get_embedder()
        reindexed = 0
        for row in rows:
            stored_path = Path(row["source_path"])
            if not stored_path.exists():
                continue
            segments = extract_document_segments(stored_path)
            chunk_rows = []
            for segment in segments:
                for chunk in chunk_text(segment["text"]):
                    chunk_rows.append(
                        {
                            "content": chunk,
                            "page_label": segment["page_label"],
                        }
                    )
            embeddings = embedder.encode(item["content"] for item in chunk_rows)
            for index, item in enumerate(chunk_rows):
                connection.execute(
                    """
                    INSERT INTO chunks(document_id, chunk_index, content, page_label, embedding)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        row["id"],
                        index,
                        item["content"],
                        item["page_label"],
                        serialize_vector(embeddings[index]),
                    ),
                )
            reindexed += 1

    return reindexed
