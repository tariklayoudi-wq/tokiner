import os
from fastapi.testclient import TestClient
from free_cred.api import create_app
from free_cred.providers import MockProvider


def test_health():
    app = create_app(providers=[MockProvider("p1")])
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_route_positive():
    p1 = MockProvider("p1")
    app = create_app(providers=[p1])
    client = TestClient(app)
    r = client.post("/route", json={"prompt": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert body["provider_id"] == "p1"
    assert "result" in body


def test_route_fallback():
    # first provider fails, second succeeds
    p1 = MockProvider("a", fail=True)
    p2 = MockProvider("b")
    app = create_app(providers=[p1, p2])
    client = TestClient(app)
    r = client.post("/route", json={"prompt": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["provider_id"] == "b"


def test_503_when_no_provider_available():
    p1 = MockProvider("a", fail=True)
    p2 = MockProvider("b", fail=True)
    app = create_app(providers=[p1, p2])
    client = TestClient(app)
    r = client.post("/route", json={"prompt": "hi"})
    assert r.status_code == 503
    body = r.json()
    assert "error" in body and "message" in body["error"]


def test_no_token_in_response():
    p1 = MockProvider("a")
    app = create_app(providers=[p1])
    client = TestClient(app)
    r = client.post("/route", json={"prompt": "hi"})
    assert r.status_code == 200
    body = r.json()
    # ensure no 'token' field anywhere in serialized body
    def search(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() == "token":
                    return True
                if search(v):
                    return True
        elif isinstance(obj, list):
            for it in obj:
                if search(it):
                    return True
        return False

    assert not search(body)
