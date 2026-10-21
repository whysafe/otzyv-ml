import json

from otzyv.__main__ import main
from otzyv.train import run


def test_run_writes_reports_and_model(tmp_path):
    res = run(out_model=tmp_path / "m.joblib", reports=tmp_path / "rep")
    assert res["best"] in {"logreg", "svm"}
    assert res["holdout"]["f1_macro"] > 0.85
    assert (tmp_path / "m.joblib").exists()
    assert (tmp_path / "rep" / "confusion_matrix.png").stat().st_size > 10_000
    md = (tmp_path / "rep" / "metrics.md").read_text(encoding="utf-8")
    assert "Кросс-валидация" in md and "| спам |" in md
    data = json.loads((tmp_path / "rep" / "metrics.json").read_text(encoding="utf-8"))
    assert set(data["cv"]) == {"logreg", "svm"}


def test_cli_predict_json(model_file, capsys):
    assert main(["predict", "--model", str(model_file), "--json", "Всё супер, рекомендую!"]) == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["label"] == "positive"


def test_cli_predict_missing_model(tmp_path, capsys):
    assert main(["predict", "--model", str(tmp_path / "nope.joblib"), "текст"]) == 1
    assert "Модель не найдена" in capsys.readouterr().err
