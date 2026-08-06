from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import Cookie, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .auth import authenticate_user, create_session, create_user, delete_session, get_user_by_session
from .db import init_db
from .documents import (
    SUPPORTED_SUFFIXES,
    delete_document,
    ingest_document,
    list_documents,
    reindex_user_documents,
    save_upload,
)
from .embeddings import get_embedder
from .qa import answer_question


app = FastAPI(title="Offline Multi-Document QA")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


@app.on_event("startup")
def startup() -> None:
    init_db()


def current_user(session_token: Optional[str]) -> Optional[dict]:
    if not session_token:
        return None
    user = get_user_by_session(session_token)
    if not user:
        return None
    return {"id": user.id, "username": user.username}


@app.get("/", response_class=HTMLResponse)
def home(request: Request, session_token: Optional[str] = Cookie(default=None)):
    user = current_user(session_token)
    documents = list_documents(user["id"]) if user else []
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "user": user,
            "documents": documents,
            "embedder_fallback": get_embedder().uses_fallback,
            "answer_result": None,
            "error": None,
            "question": "",
            "display_question": "",
        },
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.post("/register")
def register(username: str = Form(...), password: str = Form(...)):
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    try:
        user = create_user(username.strip(), password)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Username already exists.") from exc
    response = RedirectResponse("/", status_code=303)
    response.set_cookie("session_token", create_session(user.id), httponly=True, samesite="lax")
    return response


@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    user = authenticate_user(username.strip(), password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    response = RedirectResponse("/", status_code=303)
    response.set_cookie("session_token", create_session(user.id), httponly=True, samesite="lax")
    return response


@app.post("/logout")
def logout(session_token: Optional[str] = Cookie(default=None)):
    if session_token:
        delete_session(session_token)
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie("session_token")
    return response


@app.post("/upload", response_class=HTMLResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    session_token: Optional[str] = Cookie(default=None),
):
    user = current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Login required.")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Supported types: {', '.join(sorted(SUPPORTED_SUFFIXES))}")
    stored_path = save_upload(user["id"], file.filename, file.file)
    ingest_document(user["id"], file.filename, stored_path)
    return RedirectResponse("/", status_code=303)


@app.post("/reindex")
def reindex_documents(session_token: Optional[str] = Cookie(default=None)):
    user = current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Login required.")
    reindex_user_documents(user["id"])
    return RedirectResponse("/", status_code=303)


@app.post("/documents/{document_id}/delete")
def remove_document(document_id: int, session_token: Optional[str] = Cookie(default=None)):
    user = current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Login required.")
    delete_document(user["id"], document_id)
    return RedirectResponse("/", status_code=303)


@app.post("/ask", response_class=HTMLResponse)
async def ask_question(
    request: Request,
    question: str = Form(...),
    session_token: Optional[str] = Cookie(default=None),
):
    user = current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Login required.")
    documents = list_documents(user["id"])
    result = answer_question(user["id"], question.strip())
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "user": user,
            "documents": documents,
            "embedder_fallback": get_embedder().uses_fallback,
            "answer_result": result,
            "error": None,
            "question": "",
            "display_question": question.strip(),
        },
    )
