"""CLI: python -m otzyv {train,predict,serve}."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .model import ROOT, Classifier, model_path


def _cmd_train(args) -> int:
    from .train import run

    res = run(out_model=Path(args.model) if args.model else None,
              screenshot=ROOT / "screenshot.png" if args.screenshot else None)
    for kind, v in res["cv"].items():
        print(f"CV {kind:6}  accuracy {v['accuracy'][0]:.3f} ± {v['accuracy'][1]:.3f}  "
              f"f1_macro {v['f1_macro'][0]:.3f} ± {v['f1_macro'][1]:.3f}")
    h = res["holdout"]
    print(f"Лучшая: {res['best']}, holdout accuracy {h['accuracy']:.3f}, f1_macro {h['f1_macro']:.3f}")
    print(f"Модель: {res['model']}\nОтчёт: reports/metrics.md, reports/confusion_matrix.png")
    return 0


def _cmd_predict(args) -> int:
    path = Path(args.model) if args.model else model_path()
    if not path.exists():
        print(f"Модель не найдена: {path}. Сначала: python -m otzyv train", file=sys.stderr)
        return 1
    clf = Classifier.load(path)
    texts = args.text or [line.strip() for line in sys.stdin if line.strip()]
    for p in clf.predict(texts, top=args.top):
        if args.json:
            print(json.dumps(p.as_dict(), ensure_ascii=False))
            continue
        words = ", ".join(f"{w['word']} ({w['weight']:+.2f})" for w in p.top_words) or "—"
        print(f"[{p.label_ru} {p.confidence:.0%}] {p.text}\n    слова: {words}")
    return 0


def _cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("otzyv.api:app", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="otzyv", description="Тональность и спам в русских отзывах")
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="кросс-валидация, отчёт, сохранение модели")
    t.add_argument("--model", help="куда сохранить модель (по умолчанию models/model.joblib)")
    t.add_argument("--screenshot", action="store_true", help="также сохранить график в screenshot.png")
    t.set_defaults(func=_cmd_train)

    p = sub.add_parser("predict", help="классифицировать тексты (аргументы или stdin)")
    p.add_argument("text", nargs="*")
    p.add_argument("--model")
    p.add_argument("--top", type=int, default=5, help="сколько слов в объяснении")
    p.add_argument("--json", action="store_true", help="вывод в JSON Lines")
    p.set_defaults(func=_cmd_predict)

    s = sub.add_parser("serve", help="запустить FastAPI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
