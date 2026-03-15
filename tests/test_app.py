from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from case_web.app import create_app


class StubInferenceService:
    def __init__(self):
        self.runtime_dir = Path(tempfile.mkdtemp())
        self.UPLOADS_DIR = self.runtime_dir / "uploads"
        self.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        self.saved_payload = {
            "ready": True,
            "artifacts_dir": str(self.runtime_dir),
            "model": {
                "num_classes": 20,
                "class_names": ["class_0", "class_1"],
                "validation_accuracy": 0.12,
                "validation_loss": 2.3,
            },
            "training": {
                "epochs": [1, 2],
                "train_accuracy_curve": [0.1, 0.2],
                "val_accuracy_curve": [0.09, 0.12],
                "train_loss_curve": [3.0, 2.7],
                "val_loss_curve": [3.1, 2.8],
                "train_class_counts": {"0": 10, "1": 11},
                "top5_validation_classes": [{"class_id": 1, "label": "class_1", "count": 11}],
            },
            "test": None,
            "plots": {},
            "downloads": {},
        }

    def ensure_runtime_dirs(self):
        self.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    def run_test_inference(self, upload_path):
        return {
            "file_name": Path(upload_path).name,
            "num_samples": 2,
            "final_accuracy": 0.5,
            "final_loss": 1.23,
            "preview_rows": [],
        }

    def build_dashboard_payload(self):
        return self.saved_payload

    def get_plot_path(self, filename):
        plot_path = self.runtime_dir / filename
        plot_path.write_text("<html></html>", encoding="utf-8")
        return plot_path

    def get_result_path(self, filename):
        result_path = self.runtime_dir / filename
        result_path.write_text("{}", encoding="utf-8")
        return result_path


class AppRoutesTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        self.stub_service = StubInferenceService()
        self.app = create_app(
            {
                "TESTING": True,
                "SECRET_KEY": "test-secret",
                "DATABASE": str(self.db_path),
                "INFERENCE_SERVICE": self.stub_service,
            }
        )
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def login_admin(self):
        return self.client.post(
            "/login",
            data={"username": "admin", "password": "admin"},
            follow_redirects=True,
        )

    def test_health_route(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"ok": True})

    def test_admin_can_create_user(self):
        self.login_admin()
        response = self.client.post(
            "/admin",
            data={
                "username": "user1",
                "password": "pass1",
                "role": "user",
                "first_name": "Иван",
                "last_name": "Иванов",
            },
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Пользователь создан", response.get_data(as_text=True))

    def test_analytics_route_returns_payload_for_logged_user(self):
        self.login_admin()
        response = self.client.get("/api/analytics")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["model"]["num_classes"], 20)

    def test_upload_npz_calls_inference_service(self):
        self.login_admin()
        fake_npz = io.BytesIO(b"PK")
        response = self.client.post(
            "/upload",
            data={"file": (fake_npz, "test.npz")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["result"]["num_samples"], 2)

    def test_db_dump_download(self):
        self.login_admin()
        response = self.client.get("/admin/db-dump")
        self.assertEqual(response.status_code, 200)
        self.assertIn("CREATE TABLE users", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
