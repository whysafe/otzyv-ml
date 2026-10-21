"""HTTP API: POST /predict с объяснением, какие слова повлияли на ответ."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import Classifier, model_path, train_full

_state: dict[str, Classifier] = {}


def get_classifier() -> Classifier:
    if "clf" not in _state:
        path = model_path()
        if path.exists():
            _state["clf"] = Classifier.load(path)
        else:
            clf = train_full()
            clf.save(path)
            _state["clf"] = clf
    return _state["clf"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_classifier()
    yield


app = FastAPI(title="otzyv-ml", description="Тональность и спам в русских отзывах", lifespan=lifespan)


class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, examples=["Доставили быстро, всё вкусно!"])
    top: int = Field(5, ge=1, le=20, description="сколько слов вернуть в объяснении")


class Word(BaseModel):
    word: str
    weight: float


class PredictOut(BaseModel):
    text: str
    label: str
    label_ru: str
    confidence: float
    scores: dict[str, float]
    top_words: list[Word]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "classes": get_classifier().classes}


@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn) -> dict:
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="пустой текст")
    return get_classifier().predict_one(body.text, top=body.top).as_dict()
