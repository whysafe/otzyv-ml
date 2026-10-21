import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(model_file, monkeypatch_module):
    monkeypatch_module.setenv("OTZYV_MODEL", str(model_file))
    from otzyv import api

    api._state.clear()
    with TestClient(api.app) as c:
        yield c
    api._state.clear()


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert sorted(r.json()["classes"]) == ["negative", "positive", "spam"]


def test_predict_spam_with_explanation(client):
    r = client.post("/predict", json={"text": "Пассивный доход от 3000 в неделю! Жми https://clck.ru/xyz", "top": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "spam"
    assert body["label_ru"] == "спам"
    assert 0 < len(body["top_words"]) <= 3
    assert set(body["scores"]) == {"negative", "positive", "spam"}


def test_predict_negative(client):
    r = client.post("/predict", json={"text": "Ужасно, товар бракованный, поддержка не отвечает"})
    assert r.json()["label"] == "negative"


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "ok", "top": 0}])
def test_validation(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_whitespace_text_rejected(client):
    assert client.post("/predict", json={"text": "   "}).status_code == 422
