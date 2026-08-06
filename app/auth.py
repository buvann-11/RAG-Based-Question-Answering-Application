from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from typing import Optional

from .db import connection_scope, get_connection


def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        200_000,
    ).hex()


@dataclass
class User:
    id: int
    username: str


def create_user(username: str, password: str) -> User:
    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)
    with connection_scope() as connection:
        cursor = connection.execute(
            """
            INSERT INTO users(username, password_hash, password_salt)
            VALUES (?, ?, ?)
            """,
            (username, password_hash, salt),
        )
        return User(id=cursor.lastrowid, username=username)


def authenticate_user(username: str, password: str) -> Optional[User]:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, username, password_hash, password_salt
            FROM users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()
    if not row:
        return None
    attempted_hash = hash_password(password, row["password_salt"])
    if attempted_hash != row["password_hash"]:
        return None
    return User(id=row["id"], username=row["username"])


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with connection_scope() as connection:
        connection.execute(
            "INSERT INTO sessions(token, user_id) VALUES (?, ?)",
            (token, user_id),
        )
    return token


def get_user_by_session(token: str) -> Optional[User]:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT users.id, users.username
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token = ?
            """,
            (token,),
        ).fetchone()
    if not row:
        return None
    return User(id=row["id"], username=row["username"])


def delete_session(token: str) -> None:
    with connection_scope() as connection:
        connection.execute("DELETE FROM sessions WHERE token = ?", (token,))
