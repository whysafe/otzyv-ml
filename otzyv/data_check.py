"""Проверка CSV-датасета: пустые тексты, неизвестные метки, дубликаты."""

from __future__ import annotations

import csv
from pathlib import Path

ALLOWED = {"positive", "negative", "spam"}


def check_dataset(path: Path | str) -> list[str]:
    problems: list[str] = []
    seen: dict[str, int] = {}
    with Path(path).open(encoding="utf-8", newline="") as f:
        for line, row in enumerate(csv.DictReader(f), start=2):
            text = (row.get("text") or "").strip()
            label = row.get("label")
            if not text:
                problems.append(f"строка {line}: пустой текст")
                continue
            if label not in ALLOWED:
                problems.append(f"строка {line}: неизвестная метка {label!r}")
            if text in seen:
                problems.append(f"строка {line}: дубликат строки {seen[text]}")
            else:
                seen[text] = line
    return problems
