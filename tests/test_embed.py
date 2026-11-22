"""Tests for /embed with the model mocked (no weights are downloaded)."""

import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# insightface may not be installed where the tests run; compare_faces only
# needs the name at import time.
try:
    import insightface.app  # noqa: F401
except Exception:
    pkg = types.ModuleType("insightface")
    mod = types.ModuleType("insightface.app")
    mod.FaceAnalysis = object
    pkg.app = mod
    sys.modules["insightface"] = pkg
    sys.modules["insightface.app"] = mod

import cv2  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app as api  # noqa: E402
import compare_faces  # noqa: E402


class FakeFace:
    def __init__(self, bbox, seed=0):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.embedding = np.random.RandomState(seed).randn(512).astype(np.float32) * 5
        self.det_score = 0.93
        self.kps = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.float32)


class FakeModel:
    def __init__(self, faces):
        self.faces = faces

    def get(self, img):
        return list(self.faces)


def jpeg(w=64, h=48):
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), 128, dtype=np.uint8))
    assert ok
    return buf.tobytes()


def use_faces(monkeypatch, faces):
    monkeypatch.setattr(compare_faces, "get_model", lambda: FakeModel(faces))


@pytest.fixture
def client(monkeypatch):
    use_faces(monkeypatch, [])
    monkeypatch.setattr(api, "API_KEY", "")
    # No `with`, so the lifespan (which loads the real model) is not run.
    return TestClient(api.app)


def test_embed_returns_normalised_largest_face(client, monkeypatch):
    small = FakeFace([0, 0, 10, 10], seed=1)
    big = FakeFace([5, 6, 45, 46], seed=2)
    use_faces(monkeypatch, [small, big])
    r = client.post("/embed", files={"image": ("a.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "buffalo_sc"
    assert body["dim"] == 512 and len(body["embedding"]) == 512
    assert abs(np.linalg.norm(body["embedding"]) - 1.0) < 1e-6
    expected = big.embedding / np.linalg.norm(big.embedding)
    assert np.allclose(body["embedding"], expected, atol=1e-5)
    assert body["faces"] == 2
    assert (body["width"], body["height"]) == (64, 48)
    assert (body["face_width"], body["face_height"]) == (40, 40)
    assert body["det_score"] == pytest.approx(0.93)
    assert len(body["kps"]) == 5 and body["kps"][0] == [1.0, 2.0]


def test_embed_no_face_is_400(client, monkeypatch):
    use_faces(monkeypatch, [])
    r = client.post("/embed", files={"image": ("a.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "no_face"


def test_embed_bad_image_is_400(client):
    r = client.post("/embed", files={"image": ("a.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "bad_image"


def test_embed_requires_key_when_configured(client, monkeypatch):
    use_faces(monkeypatch, [FakeFace([0, 0, 10, 10])])
    monkeypatch.setattr(api, "API_KEY", "secret")
    files = {"image": ("a.jpg", jpeg(), "image/jpeg")}
    assert client.post("/embed", files=files).status_code == 401
    assert client.post("/embed", files=files, headers={"X-Api-Key": "no"}).status_code == 401
    assert client.post("/embed", files=files, headers={"X-Api-Key": "secret"}).status_code == 200


def test_health_advertises_embed(client):
    assert client.get("/health").json()["embed"] is True
