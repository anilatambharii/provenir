from __future__ import annotations

from pathlib import Path

import pytest

import provenir.server.app as server_mod


class TestServerWithoutFastAPI:
    """Tests that verify graceful behaviour when FastAPI is not installed."""

    def setup_method(self) -> None:
        self._original = server_mod._HAS_FASTAPI
        server_mod._HAS_FASTAPI = False

    def teardown_method(self) -> None:
        server_mod._HAS_FASTAPI = self._original

    def test_create_app_raises_import_error(self) -> None:
        with pytest.raises(ImportError, match="FastAPI"):
            server_mod.create_app()

    def test_run_server_raises_import_error(self) -> None:
        with pytest.raises(ImportError, match="FastAPI"):
            server_mod.run_server()


class TestServerWithFastAPI:
    """Tests that exercise the app when FastAPI is available."""

    def test_create_app_returns_fastapi_app(self, tmp_path: "Path") -> None:  # type: ignore[name-defined]

        if not server_mod._HAS_FASTAPI:
            pytest.skip("fastapi not installed")

        app = server_mod.create_app(
            manifest_dir=str(tmp_path / "manifests"),
            adapter_dir=str(tmp_path / "adapters"),
            audit_dir=str(tmp_path),
        )
        assert app is not None
        assert app.title == "Provenir API"

    def test_health_route_registered(self, tmp_path: "Path") -> None:  # type: ignore[name-defined]

        if not server_mod._HAS_FASTAPI:
            pytest.skip("fastapi not installed")

        from fastapi.testclient import TestClient

        app = server_mod.create_app(
            manifest_dir=str(tmp_path / "manifests"),
            adapter_dir=str(tmp_path / "adapters"),
            audit_dir=str(tmp_path),
        )
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_train_endpoint(self, tmp_path: "Path") -> None:  # type: ignore[name-defined]

        if not server_mod._HAS_FASTAPI:
            pytest.skip("fastapi not installed")

        from fastapi.testclient import TestClient

        app = server_mod.create_app(
            manifest_dir=str(tmp_path / "manifests"),
            adapter_dir=str(tmp_path / "adapters"),
            audit_dir=str(tmp_path),
        )
        client = TestClient(app)
        payload = {
            "config": {"name": "test", "backend": "stub"},
            "records": [{"prompt": "q", "response": "a"}],
        }
        resp = client.post("/jobs/train", json=payload)
        assert resp.status_code == 200
        assert "run_id" in resp.json()

    def test_eval_endpoint(self, tmp_path: "Path") -> None:  # type: ignore[name-defined]

        if not server_mod._HAS_FASTAPI:
            pytest.skip("fastapi not installed")

        from fastapi.testclient import TestClient

        app = server_mod.create_app(
            manifest_dir=str(tmp_path / "manifests"),
            adapter_dir=str(tmp_path / "adapters"),
            audit_dir=str(tmp_path),
        )
        client = TestClient(app)
        payload = {
            "records": [{"prompt": "q", "response": "a"}],
            "predictions": ["a"],
        }
        resp = client.post("/eval", json=payload)
        assert resp.status_code == 200

    def test_list_manifests_empty(self, tmp_path: "Path") -> None:  # type: ignore[name-defined]

        if not server_mod._HAS_FASTAPI:
            pytest.skip("fastapi not installed")

        from fastapi.testclient import TestClient

        app = server_mod.create_app(
            manifest_dir=str(tmp_path / "manifests"),
            adapter_dir=str(tmp_path / "adapters"),
            audit_dir=str(tmp_path),
        )
        client = TestClient(app)
        resp = client.get("/manifests")
        assert resp.status_code == 200
        assert resp.json()["run_ids"] == []

    def test_get_nonexistent_manifest_404(self, tmp_path: "Path") -> None:  # type: ignore[name-defined]

        if not server_mod._HAS_FASTAPI:
            pytest.skip("fastapi not installed")

        from fastapi.testclient import TestClient

        app = server_mod.create_app(
            manifest_dir=str(tmp_path / "manifests"),
            adapter_dir=str(tmp_path / "adapters"),
            audit_dir=str(tmp_path),
        )
        client = TestClient(app)
        resp = client.get("/manifests/nonexistent-id")
        assert resp.status_code == 404
