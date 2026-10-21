"""Модель: TF-IDF по словам и символьным n-граммам + линейный классификатор."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .preprocess import normalize

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "reviews.csv"
DEFAULT_MODEL_PATH = ROOT / "models" / "model.joblib"
LABELS_RU = {"positive": "позитивный", "negative": "негативный", "spam": "спам"}
SERVICE_TOKENS = {"urltoken": "<ссылка>", "phonetoken": "<телефон>", "handletoken": "<@ник>", "num": "<число>"}


def model_path() -> Path:
    return Path(os.environ.get("OTZYV_MODEL", DEFAULT_MODEL_PATH))


def load_dataset(path: Path = DATA_PATH) -> tuple[list[str], list[str]]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["label"] for r in rows]


def build_pipeline(kind: str = "logreg") -> Pipeline:
    """kind: ``logreg`` (LogisticRegression) или ``svm`` (LinearSVC)."""
    features = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=normalize, analyzer="word", ngram_range=(1, 2),
                                 token_pattern=r"(?u)\b\w+\b|[!?%₽]", min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(2, 5),
                                 min_df=2, sublinear_tf=True, max_features=30000)),
    ])
    if kind == "logreg":
        clf = LogisticRegression(C=10.0, max_iter=2000)
    elif kind == "svm":
        clf = LinearSVC(C=1.0)
    else:
        raise ValueError(f"неизвестный классификатор: {kind}")
    return Pipeline([("features", features), ("clf", clf)])


@dataclass
class Prediction:
    text: str
    label: str
    confidence: float
    scores: dict[str, float]
    top_words: list[dict] = field(default_factory=list)

    @property
    def label_ru(self) -> str:
        return LABELS_RU.get(self.label, self.label)

    def as_dict(self) -> dict:
        return {"text": self.text, "label": self.label, "label_ru": self.label_ru,
                "confidence": round(self.confidence, 4),
                "scores": {k: round(v, 4) for k, v in self.scores.items()},
                "top_words": self.top_words}


class Classifier:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.features: FeatureUnion = pipeline.named_steps["features"]
        self.clf = pipeline.named_steps["clf"]
        self.classes = [str(c) for c in self.clf.classes_]
        word_vec: TfidfVectorizer = self.features.transformer_list[0][1]
        self._n_word = len(word_vec.vocabulary_)
        self._word_names = word_vec.get_feature_names_out()

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Classifier":
        return cls(joblib.load(path or model_path()))

    def save(self, path: Path | str | None = None) -> Path:
        path = Path(path or model_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        return path

    def _scores(self, X) -> np.ndarray:
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(X)
        d = self.clf.decision_function(X)
        e = np.exp(d - d.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def explain(self, X_row, class_idx: int, top: int = 5) -> list[dict]:
        """Слова (униграммы и биграммы), сильнее всего тянущие к предсказанному классу.

        Для линейной модели вклад признака = tf-idf * коэффициент класса.
        Символьные n-граммы в объяснение не попадают: они нечитаемы.
        """
        row = X_row.tocsr()[:, : self._n_word]
        coef = self.clf.coef_[class_idx, : self._n_word]
        idx = row.indices
        contrib = row.data * coef[idx]
        order = np.argsort(-contrib)
        out = []
        for i in order[:top]:
            if contrib[i] <= 0:
                break
            name = str(self._word_names[idx[i]])
            name = " ".join(SERVICE_TOKENS.get(w, w) for w in name.split())
            out.append({"word": name, "weight": round(float(contrib[i]), 4)})
        return out

    def predict(self, texts: list[str], top: int = 5) -> list[Prediction]:
        X = self.features.transform(texts)
        scores = self._scores(X)
        result = []
        for i, text in enumerate(texts):
            k = int(np.argmax(scores[i]))
            result.append(Prediction(
                text=text, label=self.classes[k], confidence=float(scores[i, k]),
                scores={c: float(s) for c, s in zip(self.classes, scores[i])},
                top_words=self.explain(X[i], k, top)))
        return result

    def predict_one(self, text: str, top: int = 5) -> Prediction:
        return self.predict([text], top)[0]


def train_full(kind: str = "logreg", data_path: Path = DATA_PATH) -> Classifier:
    texts, labels = load_dataset(data_path)
    pipe = build_pipeline(kind)
    pipe.fit(texts, labels)
    return Classifier(pipe)
