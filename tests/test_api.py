import os
from fastapi.testclient import TestClient
from free_cred.api import create_app
from free_cred.providers import MockProvider


class CapturingProvider(MockProvider):
    def __init__(self, id):
        super().__init__(id)
        self.last_args = None
        self.last_kwargs = None

    def call(self, *args, **kwargs):
        self.last_args = args
        self.last_kwargs = kwargs
        return {"provider": self.id, "content": "ok"}


class BrokenHealthProvider(MockProvider):
    def health_check(self):
        raise RuntimeError("secret-token-like-detail")


class InvalidHealthProvider(MockProvider):
    def health_check(self):
        return "not-a-dict"


def test_health():
    app = create_app(providers=[MockProvider("p1")])
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_provider_health_endpoints_include_status_and_quota():
    app = create_app(
        providers=[
            MockProvider("nvidia-nemotron", remaining=80, limit=100),
            MockProvider("openvino"),
            MockProvider("copilot", fail=True),
        ]
    )
    client = TestClient(app)

    nvidia = client.get("/health/nvidia").json()
    openvino = client.get("/health/openvino").json()
    copilot = client.get("/health/copilot").json()
    status = client.get("/providers/status").json()

    assert nvidia["ok"] is True
    assert nvidia["checks"][0]["quota"]["ratio"] == 0.8
    assert openvino["ok"] is True
    assert copilot["ok"] is False
    assert status["status"] == "healthy"
    assert [provider["provider"] for provider in status["providers"]] == [
        "nvidia-nemotron",
        "openvino",
        "copilot",
    ]


def test_provider_health_endpoint_handles_broken_health_check_without_leaking_details():
    app = create_app(providers=[BrokenHealthProvider("nvidia-nemotron")])
    client = TestClient(app)

    r = client.get("/health/nvidia")

    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    check = body["checks"][0]
    assert check["status"] == "unhealthy"
    assert check["message"] == "provider health check failed"
    assert check["error_type"] == "RuntimeError"
    assert "secret-token-like-detail" not in r.text


def test_provider_health_endpoint_normalizes_invalid_health_check_result():
    app = create_app(providers=[InvalidHealthProvider("nvidia-nemotron")])
    client = TestClient(app)

    r = client.get("/health/nvidia")

    assert r.status_code == 200
    check = r.json()["checks"][0]
    assert check["provider"] == "nvidia-nemotron"
    assert check["status"] == "unhealthy"
    assert check["ok"] is False
    assert check["message"] == "provider health check returned invalid result"


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


def test_openai_compatible_models_endpoint():
    app = create_app(providers=[MockProvider("p1")])
    client = TestClient(app)

    r = client.get("/v1/models")

    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "free-cred-router"


def test_custom_provider_discovery_compatibility_endpoints():
    app = create_app(providers=[MockProvider("p1")])
    client = TestClient(app)

    assert client.get("/api/v1/models").status_code == 200
    assert client.get("/v1/models/free-cred-router").status_code == 200
    assert client.get("/version").json()["version"] == "free-cred-router-0.0.0"
    assert client.get("/props").json()["openai_compatible"] is True
    assert client.get("/v1/props").json()["openai_compatible"] is True
    assert client.get("/api/tags").json()["models"][0]["name"] == "free-cred-router"
    assert client.post("/api/show", json={"model": "free-cred-router"}).json()["model"] == "free-cred-router"


def test_openai_compatible_chat_completion_uses_router():
    app = create_app(providers=[MockProvider("p1")])
    client = TestClient(app)

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "free-cred-router",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
        },
    )

    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "free-cred-router"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["free_cred"]["provider"] == "p1"
    assert "user: hello" in body["choices"][0]["message"]["content"]


def test_openai_compatible_chat_completion_fallback_error_shape():
    app = create_app(providers=[MockProvider("a", fail=True)])
    client = TestClient(app)

    r = client.post(
        "/v1/chat/completions",
        json={"model": "free-cred-router", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert r.status_code == 503
    body = r.json()
    assert body["error"]["type"] == "free_cred_provider_error"
    assert body["error"]["code"] == 503


def test_openai_compatible_chat_completion_stream():
    app = create_app(providers=[MockProvider("p1")])
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "free-cred-router",
            "messages": [{"role": "user", "content": "stream me"}],
            "stream": True,
        },
    ) as r:
        body = "".join(r.iter_text())

    assert r.status_code == 200
    assert "data: " in body
    assert '"object": "chat.completion.chunk"' in body
    assert '"usage":' in body
    assert "user: stream me" in body
    assert "data: [DONE]" in body


def test_create_app_loads_provider_key_file_from_env(monkeypatch, tmp_path):
    key_file = tmp_path / "nvidia_keys.txt"
    key_file.write_text("test-key\n", encoding="utf-8")
    monkeypatch.setenv("PROVIDERS", "nvidia")
    monkeypatch.setenv("FREE_CRED_KEYS_PROVIDER", "nvidia")
    monkeypatch.setenv("FREE_CRED_KEYS_FILE", str(key_file))

    app = create_app()
    client = TestClient(app)
    r = client.get("/v1/models")

    assert r.status_code == 200
    assert os.environ["NVIDIA_API_KEY_ENVS"] == "NVIDIA_API_KEY_1"


def test_openai_compatible_chat_completion_passes_tools_to_router_provider():
    provider = CapturingProvider("p1")
    app = create_app(providers=[provider])
    client = TestClient(app)
    tools = [{"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}]

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "free-cred-router",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        },
    )

    assert r.status_code == 200
    assert provider.last_kwargs["tools"] == tools
    assert provider.last_kwargs["tool_choice"] == "auto"
    assert provider.last_kwargs["parallel_tool_calls"] is False
    assert provider.last_args[0]["messages"] == [{"role": "user", "content": "hello"}]
