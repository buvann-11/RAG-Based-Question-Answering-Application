from __future__ import annotations

import hashlib
import html
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from app import db
from app.documents import list_documents
from app.main import app


POLICY_TEXT = """
SECTION B) DEFINITIONS - STANDARD DEFINITIONS
1. Accident:- An Accident means sudden, unforeseen and involuntary event caused by external, visible and violent means.

SECTION C) BENEFITS COVERED UNDER THE POLICY
PART A- COVERAGE- Domestic (Within India Only)
I. IN-PATIENT BENEFITS FOR DOMESTIC COVER
1. In-patient Hospitalization Treatment
If You are advised Hospitalization within India by a Medical Practitioner, then We will pay You, Reasonable and Customary Medical Expenses incurred subject to
i. Room rent and Boarding expenses as provided by the Hospital/Nursing Home without any sub limit
ii. If admitted in ICU, the Company will pay up to actual ICU expenses provided by Hospital.
iii. Nursing Expenses as provided by the Hospital
iv. Surgeon, Anesthetist, Medical Practitioner, Consultants, Specialists Fees.
v. Anesthesia, Blood, Oxygen, Operation Theatre Charges, surgical appliances.
vi. Dialysis, Chemotherapy, Radiotherapy, Physiotherapy.
vii. Medicines & Drugs.

2. Pre-Hospitalization
The Medical Expenses incurred during the 60 days immediately before You were Hospitalized, provided that:
Such Medical Expenses were incurred for the same Illness/Injury for which subsequent Hospitalization was required, and
We have accepted an Inpatient Hospitalization claim under Inpatient Hospitalization Treatment.
""".strip()


class FakeEmbedder:
    uses_fallback = True

    def encode(self, texts):
        vectors = []
        for text in texts:
            vector = np.zeros(48, dtype=float)
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = digest[0] % vector.size
                vector[bucket] += 1.0
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return np.vstack(vectors) if vectors else np.zeros((0, 48), dtype=float)


class AppTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db_path = self.root / "test.sqlite3"
        self.upload_dir = self.root / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.fake_embedder = FakeEmbedder()

        self.patches = [
            patch("app.db.DB_PATH", self.db_path),
            patch("app.documents.UPLOAD_DIR", self.upload_dir),
            patch("app.documents.get_embedder", return_value=self.fake_embedder),
            patch("app.retrieval.get_embedder", return_value=self.fake_embedder),
            patch("app.main.get_embedder", return_value=self.fake_embedder),
            patch("app.qa.generate_grounded_answer", return_value=None),
        ]
        for active_patch in self.patches:
            active_patch.start()

        db.init_db()
        self.client = TestClient(app)
        self.register_and_login("tester", "secret123")

    def tearDown(self) -> None:
        self.client.close()
        for active_patch in reversed(self.patches):
            active_patch.stop()
        try:
            self.temp_dir.cleanup()
        except PermissionError:
            pass

    def register_and_login(self, username: str, password: str) -> None:
        response = self.client.post(
            "/register",
            data={"username": username, "password": password},
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)

    def upload_policy_text(self, filename: str = "policy.txt") -> None:
        response = self.client.post(
            "/upload",
            files={"file": (filename, BytesIO(POLICY_TEXT.encode("utf-8")), "text/plain")},
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)

    def test_definition_question_returns_exact_definition(self) -> None:
        self.upload_policy_text()
        response = self.client.post(
            "/ask",
            data={"question": "What is the definition of an Accident in the policy"},
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("sudden, unforeseen and involuntary event", response.text)
        self.assertIn("[1]", response.text)

    def test_pre_hospitalization_question_returns_correct_clause(self) -> None:
        self.upload_policy_text()
        response = self.client.post(
            "/ask",
            data={"question": "The Medical Expenses incurred during the Pre-Hospitalization"},
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("60 days immediately before", response.text)
        self.assertIn("Inpatient Hospitalization claim", response.text)

    def test_coverage_question_returns_inpatient_expense_list(self) -> None:
        self.upload_policy_text()
        response = self.client.post(
            "/ask",
            data={"question": "What expenses are covered under In-patient Hospitalization Treatment in India"},
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        rendered = html.unescape(response.text)
        self.assertIn("Room rent and Boarding expenses", rendered)
        self.assertIn("If admitted in ICU", rendered)
        self.assertIn("Medicines & Drugs", rendered)

    def test_unrelated_question_hides_citations_and_context(self) -> None:
        self.upload_policy_text()
        response = self.client.post(
            "/ask",
            data={"question": "when will ipl 2026 start"},
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        rendered = html.unescape(response.text)
        self.assertIn("I couldn't find the answer in the uploaded document.", rendered)
        self.assertNotIn("Retrieved Context", rendered)
        self.assertNotIn("Citations", rendered)

    def test_delete_document_removes_record(self) -> None:
        self.upload_policy_text("delete-me.txt")
        documents = list_documents(1)
        self.assertEqual(len(documents), 1)

        response = self.client.post(f"/documents/{documents[0]['id']}/delete", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(list_documents(1), [])

    def test_unsupported_file_upload_is_rejected(self) -> None:
        response = self.client.post(
            "/upload",
            files={"file": ("bad.exe", BytesIO(b"not allowed"), "application/octet-stream")},
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 400)

    def test_reindex_route_redirects(self) -> None:
        self.upload_policy_text()
        response = self.client.post("/reindex", follow_redirects=False)
        self.assertEqual(response.status_code, 303)

    def test_enter_to_send_script_is_present(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("requestSubmit()", response.text)
        self.assertIn("event.key === \"Enter\"", response.text)


if __name__ == "__main__":
    unittest.main()
