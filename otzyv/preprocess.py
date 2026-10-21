"""Нормализация русского текста перед векторизацией."""

from __future__ import annotations

import re

URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+\.(?:ru|com|xyz|net|org|ly|me|рф|io)(?:/\S*)?", re.IGNORECASE)
HANDLE_RE = re.compile(r"(?<!\w)@\w{3,}")
PHONE_RE = re.compile(r"(?:\+7|\b8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}\b")
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
REPEAT_RE = re.compile(r"(\w)\1{2,}")
JUNK_RE = re.compile(r"[^\w\s!?%₽]")
SPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Приводит текст к виду, удобному для TF-IDF.

    Ссылки, телефоны и @-ники заменяются служебными токенами (это главный
    признак спама, а конкретный адрес не важен), ё -> е, «оооочень» -> «оочень»,
    числа -> ``num``, лишняя пунктуация убирается, но ``!``, ``?``, ``%`` и ``₽``
    остаются: они несут сигнал.
    """
    text = text.lower().replace("ё", "е")
    text = URL_RE.sub(" urltoken ", text)
    text = PHONE_RE.sub(" phonetoken ", text)
    text = HANDLE_RE.sub(" handletoken ", text)
    text = NUM_RE.sub(" num ", text)
    text = REPEAT_RE.sub(r"\1\1", text)
    text = JUNK_RE.sub(" ", text)
    text = re.sub(r"([!?%₽])", r" \1 ", text)
    return SPACE_RE.sub(" ", text).strip()
