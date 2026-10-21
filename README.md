# otzyv-ml: тональность и спам в русских отзывах

Классификатор коротких русских текстов на три класса: позитивный отзыв, негативный отзыв, спам. TF-IDF по словам и символьным n-граммам плюс линейная модель, CLI и FastAPI-эндпоинт `/predict`, который показывает, какие слова повлияли на ответ.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=flat&logo=pytest&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat&logo=githubactions&logoColor=white)

![Скриншот](screenshot.png)

## Возможности

- Три класса: `positive`, `negative`, `spam`, с вероятностями по каждому.
- Объяснение ответа: топ слов и биграмм с наибольшим вкладом (tf-idf × коэффициент класса).
- Предобработка под спам: ссылки, телефоны и @-ники заменяются служебными токенами, `ё` → `е`, «оооочень» → «оочень», числа → `<число>`.
- Сравнение двух моделей на 5-фолдовой кросс-валидации, лучшая по F1 macro идёт в прод.
- Отчёт: `reports/metrics.md` (таблицы, матрица ошибок, список ошибок модели), `reports/confusion_matrix.png`, `reports/metrics.json`.
- Модель сохраняется в `joblib`, путь задаётся переменной `OTZYV_MODEL`.
- CLI для текстов из аргументов или stdin, вывод в виде текста или JSON Lines.

## Метрики

Датасет 660 текстов (по 220 на класс), отложенная выборка 25%.

| Модель | CV accuracy | CV F1 macro |
|---|---|---|
| TF-IDF + LogisticRegression (выбрана) | 0.965 ± 0.014 | 0.965 ± 0.014 |
| TF-IDF + LinearSVC | 0.965 ± 0.014 | 0.965 ± 0.014 |

| Класс (отложенная выборка) | Precision | Recall | F1 |
|---|---|---|---|
| позитивный | 0.914 | 0.964 | 0.938 |
| негативный | 0.962 | 0.909 | 0.935 |
| спам | 0.982 | 0.982 | 0.982 |
| **accuracy** | | | **0.952** |

Ошибается модель в основном на сарказме («Спасибо, конечно, что вообще доставили…») и на отзывах без оценочных слов («Обещали два дня, везли две недели»). Полный список в [reports/metrics.md](reports/metrics.md).

## Данные

**Датасет синтетический.** `data/reviews.csv` собран скриптом `data/generate.py` из шаблонов по восьми областям (электроника, доставка еды, отели, салоны, онлайн-курсы, такси, одежда, автосервис) и дополнен вручную написанными трудными примерами: отрицания («не пожалел»), сарказм, смешанные отзывы, спам без ссылок. Генерация детерминирована (seed 42), CI проверяет, что CSV совпадает с выводом генератора. На реальных отзывах метрики будут ниже: для продакшена модель нужно переобучить на размеченных живых данных, формат CSV тот же (`text,label`).

## Стек

Python 3.11+, scikit-learn (TfidfVectorizer, FeatureUnion, LogisticRegression, LinearSVC), joblib, matplotlib, FastAPI, uvicorn, pytest, GitHub Actions.

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python -m otzyv train --screenshot
```

Предсказание из командной строки:

```bash
python -m otzyv predict "Доставили быстро, пицца горячая, спасибо!" "Заработок от 5000 в день, пиши @money_x"
```

```
[позитивный 84%] Доставили быстро, пицца горячая, спасибо!
    слова: спасибо (+0.29), доставили (+0.09), горячая (+0.09), ! (+0.01)
[спам 99%] Заработок от 5000 в день, пиши @money_x
    слова: <число> (+0.23), день (+0.15), пиши (+0.15), от (+0.10), от <число> (+0.10)
```

Флаги: `--json` (JSON Lines), `--top N` (сколько слов в объяснении), `--model путь`. Без аргументов тексты читаются из stdin, по одному на строку.

API:

```bash
python -m otzyv serve --port 8000
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "Вы выиграли iPhone! Забирайте приз на https://win-prize.xyz", "top": 3}'
```

```json
{
  "label": "spam",
  "label_ru": "спам",
  "confidence": 0.9837,
  "scores": {"negative": 0.0069, "positive": 0.0094, "spam": 0.9837},
  "top_words": [{"word": "!", "weight": 0.2003}, {"word": "<ссылка>", "weight": 0.187}, {"word": "приз", "weight": 0.1509}]
}
```

Документация Swagger: `http://localhost:8000/docs`. Если файла модели нет, API обучит её при старте и сохранит.

## Как устроено

- `otzyv/preprocess.py`: нормализация текста регулярками. Она подключена как `preprocessor` в оба векторизатора и сохраняется вместе с моделью.
- `otzyv/model.py`: `FeatureUnion` из двух `TfidfVectorizer`: слова и биграммы (служебные токены и `! ? % ₽` тоже считаются словами) и символьные n-граммы 2–5 (`char_wb`, устойчивы к опечаткам и формам слов). Класс `Classifier` даёт вероятности (у LinearSVC через softmax от `decision_function`) и объяснение: для предсказанного класса берутся словарные признаки текста, их tf-idf умножается на коэффициенты модели, положительные вклады сортируются. Символьные n-граммы в объяснение не попадают, их нельзя прочитать.
- `otzyv/train.py`: `StratifiedKFold` на 5 фолдов для обеих моделей, выбор лучшей по F1 macro, отчёт на отложенных 25%, затем обучение на всём датасете и сохранение.
- `otzyv/api.py`: FastAPI, модель грузится один раз в `lifespan`, Pydantic проверяет вход (пустой текст, `top` от 1 до 20).
- `otzyv/data_check.py`: проверка CSV на пустые строки, неизвестные метки и дубликаты.

## Тесты

```bash
pytest -q
```

33 теста: предобработка (ссылки, телефоны, ники, числа, повторы букв), предсказания на очевидных примерах всех классов, сумма вероятностей, объяснение, сохранение и загрузка модели, вариант с LinearSVC, обучение с отчётами, CLI и API (`/health`, `/predict`, валидация). CI на Python 3.11 и 3.13 дополнительно проверяет воспроизводимость датасета и прогоняет обучение из CLI.

## Автор

whysafe · [whysafe.ru](https://whysafe.ru) · Telegram [@yawhysafe](https://t.me/yawhysafe) · канал [@whysafe_official](https://t.me/whysafe_official) · [Kwork](https://kwork.ru/user/whysafe) · hello@whysafe.ru
