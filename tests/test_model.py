import csv

import pytest

from otzyv.data_check import check_dataset
from otzyv.model import DATA_PATH, Classifier, build_pipeline, load_dataset

POSITIVE = [
    "Отличный магазин, доставили быстро, всё понравилось. Рекомендую!",
    "Мастер золотые руки, очень довольна результатом, спасибо!",
    "Номер чистый, персонал вежливый, обязательно вернёмся.",
]
NEGATIVE = [
    "Ужасный сервис, заказ потеряли, деньги не вернули. Не рекомендую.",
    "Курьер опоздал на два часа, еда холодная, кошмар.",
    "Качество отвратительное, развалилось через неделю, верните деньги.",
]
SPAM = [
    "Заработок от 5000 рублей в день без вложений! Пиши в телеграм @dohod_777",
    "Вы выиграли iPhone! Переходи по ссылке https://bit.ly/win123",
    "Займ без отказов, звони +7 912 345-67-89",
]


def test_dataset_balanced_and_valid():
    texts, labels = load_dataset()
    assert len(texts) >= 600
    assert set(labels) == {"positive", "negative", "spam"}
    assert check_dataset(DATA_PATH) == []


@pytest.mark.parametrize("text,label", [(t, "positive") for t in POSITIVE]
                         + [(t, "negative") for t in NEGATIVE] + [(t, "spam") for t in SPAM])
def test_obvious_examples(classifier, text, label):
    pred = classifier.predict_one(text)
    assert pred.label == label, pred.as_dict()


def test_scores_sum_to_one(classifier):
    p = classifier.predict_one(POSITIVE[0])
    assert abs(sum(p.scores.values()) - 1) < 1e-6
    assert p.confidence == max(p.scores.values())


def test_explanation_has_readable_words(classifier):
    p = classifier.predict_one(SPAM[1], top=5)
    words = [w["word"] for w in p.top_words]
    assert 0 < len(words) <= 5
    assert all(w["weight"] > 0 for w in p.top_words)
    assert any("<ссылка>" in w for w in words)
    weights = [w["weight"] for w in p.top_words]
    assert weights == sorted(weights, reverse=True)


def test_save_and_load_roundtrip(classifier, model_file):
    loaded = Classifier.load(model_file)
    a = classifier.predict(NEGATIVE)
    b = loaded.predict(NEGATIVE)
    assert [x.label for x in a] == [x.label for x in b]


def test_svm_variant_trains(tmp_path):
    texts, labels = load_dataset()
    clf = Classifier(build_pipeline("svm").fit(texts, labels))
    p = clf.predict_one(SPAM[0])
    assert p.label == "spam"
    assert abs(sum(p.scores.values()) - 1) < 1e-6


def test_unknown_kind():
    with pytest.raises(ValueError):
        build_pipeline("forest")


def test_check_dataset_reports_problems(tmp_path):
    bad = tmp_path / "bad.csv"
    with bad.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows([["ok", "positive"], ["", "negative"], ["dup", "spam"], ["dup", "spam"], ["x", "neutral"]])
    problems = check_dataset(bad)
    assert len(problems) == 3
