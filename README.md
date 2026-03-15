# PredProf Zakl

Проект состоит из двух рабочих частей:

- `case_web/` — Flask-приложение с авторизацией, ролями, загрузкой `test.npz`, аналитикой и дампом SQLite-базы.
- `notebooks/` — ноутбуки для EDA и обучения модели.

## Что уже есть

- модель в формате `h5`
- загрузка и проверка `test.npz`
- интерфейс с графиками в стиле `#00BCD4 / #000066`
- SQLite-база пользователей
- дамп БД в `case_web/space_auth_dump.sql`
- CLI для парольного архива `Answers_reduced.zip`

## Структура

```text
.
├── case_web/
│   ├── app.py
│   ├── inference.py
│   ├── artifacts/
│   ├── templates/
│   ├── space_auth.db
│   └── space_auth_dump.sql
├── data/
│   ├── Data.npz
│   └── extracted_audio/
├── notebooks/
│   ├── main_eda_extracted.ipynb
│   └── tf_audio_tiny_cnn_one_cell_colab.ipynb
├── main.ipynb
├── run_test_archive.py
└── requirements.txt
```

## Быстрый запуск

### 1. Создать окружение

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Проверить артефакты модели

Веб читает артефакты из `case_web/artifacts/`.

Минимально должны лежать:

- `case_web/artifacts/tiny_cnn_model.h5`
- `case_web/artifacts/tiny_cnn_config.json`
- `case_web/artifacts/tiny_cnn_label_mapping.csv`

Если `tiny_cnn_model.h5` лежит в корне проекта, инференс тоже его подхватит.

### 3. Запустить веб

```bash
source .venv/bin/activate
python -m flask --app case_web.app run --debug
```

Или одной командой:

```bash
./run_web.sh
```

Откройте:

- `http://127.0.0.1:5000/login`

Стандартный админ:

- логин: `admin`
- пароль: `admin`

## Как пользоваться вебом

### Администратор

- заходит под `admin/admin`
- создает нового пользователя
- при необходимости скачивает дамп БД:
  - `http://127.0.0.1:5000/admin/db-dump`

### Пользователь

- заходит под созданным логином
- на странице `/user` загружает `test.npz`
- получает:
  - `accuracy`
  - `loss`
  - confidence по тестовым записям
  - распределение train-классов
  - top-5 validation-классов
  - таблицу первых предсказаний

## Как запускать тест жюри

Жюри выдаст пароль к `Answers_reduced.zip`. Внутри лежит `Answers_reduced.npz`.

Есть два варианта.

### Вариант 1. Через терминал

```bash
source .venv/bin/activate
python run_test_archive.py --zip-path Answers_reduced.zip --password ВАШ_ПАРОЛЬ
```

После этого результаты сохраняются в:

- `case_web/runtime/results/latest_test_summary.json`
- `case_web/runtime/results/latest_test_predictions.csv`
- `case_web/runtime/plots/latest_test_confidence.html`
- `case_web/runtime/plots/latest_test_confusion_matrix.html`

### Вариант 2. Через веб

1. Распаковать архив паролем.
2. Получить файл `Answers_reduced.npz`.
3. Загрузить его на странице `/user`.

## Ноутбуки

- `notebooks/main_eda_extracted.ipynb` — только EDA-графики
- `notebooks/tf_audio_tiny_cnn_one_cell_colab.ipynb` — обучение tiny CNN и сохранение артефактов
- `main.ipynb` — исходный ноутбук, из которого была вынесена EDA-часть

## Тесты

После установки зависимостей:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

## Важное

- большие бинарники (`Data.npz`, `Answers_reduced.zip`, `h5`) в git не коммитятся
- если хотите поменять путь к артефактам модели, можно выставить переменную:

```bash
export MODEL_ARTIFACTS_DIR=/путь/к/артефактам
```
