"""Обучение и оценка: кросс-валидация двух моделей, отчёт на отложенной выборке."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,  # noqa: E402
                             f1_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split  # noqa: E402

from .model import DATA_PATH, LABELS_RU, ROOT, Classifier, build_pipeline, load_dataset, model_path  # noqa: E402

REPORTS = ROOT / "reports"
KINDS = {"logreg": "TF-IDF + LogisticRegression", "svm": "TF-IDF + LinearSVC"}
ORDER = ["positive", "negative", "spam"]


def cross_val(texts, labels, folds: int = 5, seed: int = 42) -> dict[str, dict]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    out = {}
    for kind in KINDS:
        res = cross_validate(build_pipeline(kind), texts, labels, cv=cv, scoring=["accuracy", "f1_macro"])
        out[kind] = {m: (float(np.mean(res[f"test_{m}"])), float(np.std(res[f"test_{m}"])))
                     for m in ("accuracy", "f1_macro")}
    return out


def plot_confusion(cm: np.ndarray, report: dict, title: str, path: Path) -> None:
    names = [LABELS_RU[c] for c in ORDER]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.8, 5.6), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.patch.set_facecolor("white")
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(ORDER)):
        for j in range(len(ORDER)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=13,
                    color="white" if norm[i, j] > 0.55 else "#1f2937")
    ax.set_xticks(range(3), names, fontsize=11)
    ax.set_yticks(range(3), names, fontsize=11)
    ax.set_xlabel("Предсказано", fontsize=11)
    ax.set_ylabel("Истинный класс", fontsize=11)
    ax.set_title("Матрица ошибок (отложенная выборка)", fontsize=13, pad=12)

    metrics = ["precision", "recall", "f1-score"]
    ru = {"precision": "precision", "recall": "recall", "f1-score": "F1"}
    colors = ["#93c5fd", "#3b82f6", "#1e3a8a"]
    x = np.arange(len(ORDER))
    for k, m in enumerate(metrics):
        vals = [report[c][m] for c in ORDER]
        bars = bx.bar(x + (k - 1) * 0.26, vals, 0.24, label=ru[m], color=colors[k])
        for b, v in zip(bars, vals):
            bx.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    bx.set_xticks(x, names, fontsize=11)
    bx.set_ylim(0, 1.25)
    bx.spines[["top", "right"]].set_visible(False)
    bx.legend(loc="upper center", ncol=3, frameon=False)
    bx.set_title("Метрики по классам", fontsize=13, pad=12)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run(data_path: Path = DATA_PATH, out_model: Path | None = None, reports: Path = REPORTS,
        seed: int = 42, screenshot: Path | None = None) -> dict:
    texts, labels = load_dataset(data_path)
    cv = cross_val(texts, labels, seed=seed)
    best = max(cv, key=lambda k: cv[k]["f1_macro"][0])

    X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.25, stratify=labels, random_state=seed)
    pipe = build_pipeline(best).fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    report = classification_report(y_te, pred, labels=ORDER, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, pred, labels=ORDER)
    holdout = {"accuracy": accuracy_score(y_te, pred), "f1_macro": f1_score(y_te, pred, average="macro")}

    reports.mkdir(parents=True, exist_ok=True)
    png = reports / "confusion_matrix.png"
    plot_confusion(cm, report, f"{KINDS[best]} · accuracy {holdout['accuracy']:.3f}", png)
    if screenshot:
        plot_confusion(cm, report, f"{KINDS[best]} · accuracy {holdout['accuracy']:.3f}", screenshot)
    errors = [(t, y, p) for t, y, p in zip(X_te, y_te, pred) if y != p]
    (reports / "metrics.md").write_text(
        _markdown(cv, best, report, holdout, cm, errors, len(texts), len(X_te)), encoding="utf-8")
    (reports / "metrics.json").write_text(json.dumps(
        {"cv": cv, "best": best, "holdout": holdout, "report": report}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    final = Classifier(build_pipeline(best).fit(texts, labels))
    saved = final.save(out_model or model_path())
    return {"cv": cv, "best": best, "holdout": holdout, "model": str(saved)}


def _markdown(cv, best, report, holdout, cm, errors, n_total, n_test) -> str:
    lines = ["# Метрики модели", "", f"Датасет: {n_total} текстов, отложенная выборка: {n_test} (25%, стратифицированно).", "",
             "## Кросс-валидация (5 фолдов)", "", "| Модель | Accuracy | F1 macro |", "|---|---|---|"]
    for k, v in cv.items():
        mark = " **(выбрана)**" if k == best else ""
        lines.append(f"| {KINDS[k]}{mark} | {v['accuracy'][0]:.3f} ± {v['accuracy'][1]:.3f} "
                     f"| {v['f1_macro'][0]:.3f} ± {v['f1_macro'][1]:.3f} |")
    lines += ["", f"## Отложенная выборка: {KINDS[best]}", "",
              f"Accuracy **{holdout['accuracy']:.3f}**, F1 macro **{holdout['f1_macro']:.3f}**.", "",
              "| Класс | Precision | Recall | F1 | Примеров |", "|---|---|---|---|---|"]
    for c in ORDER:
        r = report[c]
        lines.append(f"| {LABELS_RU[c]} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support'])} |")
    lines += ["", "### Матрица ошибок", "", "| истинный \\ предсказанный | " + " | ".join(LABELS_RU[c] for c in ORDER) + " |",
              "|---|---|---|---|"]
    for i, c in enumerate(ORDER):
        lines.append(f"| {LABELS_RU[c]} | " + " | ".join(str(v) for v in cm[i]) + " |")
    lines += ["", "![Матрица ошибок](confusion_matrix.png)", "", "### Ошибки модели", ""]
    if errors:
        lines += [f"- «{t}» — истинный: {LABELS_RU[y]}, предсказан: {LABELS_RU[p]}" for t, y, p in errors]
    else:
        lines.append("Ошибок на отложенной выборке нет.")
    return "\n".join(lines) + "\n"
