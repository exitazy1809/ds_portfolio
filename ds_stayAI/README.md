# SmartScore StayAI — скоринг объявлений аренды по высокому спросу

Портфолио-проект: бинарная классификация объявлений краткосрочной аренды (`is_high_demand`) и production-style batch inference в Apache Airflow (PostgreSQL → S3 → CatBoost → PostgreSQL).

> Проект учебный. Модель не использовалась в коммерческой эксплуатации. Инференс выполняется на фиксированном тестовом срезе `SmartScore_test_snapshot`.

Связанные документы:
* [`docs/TECHNICAL_AUDIT.md`](docs/TECHNICAL_AUDIT.md) — аудит: найденные дефекты, риски и статус исправлений.
* [`docs/INTERVIEW_GUIDE.md`](docs/INTERVIEW_GUIDE.md) — разбор проекта и вопросы для собеседования.
* [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md) — приоритеты доработки.
* [`dag.md`](dag.md) — описание задач DAG.

---

## 1. Бизнес-контекст и ML-задача

Платформа StayAI (учебный кейс) хочет ранжировать объявления в выдаче с учётом их потенциального спроса. Проблемы из постановки (`ds_base_final_project.md`, раздел 0): неактивные объявления занимают выдачу, а новые объявления без отзывов уходят в конец списка.

**ML-задача:** по атрибутам объявления, хоста и отзывов предсказать вероятность того, что объявление относится к сегменту высокого спроса. Скор (`score` ∈ [0, 1]) задуман как фоновый сигнал ранжирования; пересчёт пакетный.

## 2. Целевая переменная

Считается в `jupyter/EDA.ipynb` (раздел 3) по формуле из постановки. Сегмент — `city + room_type`, медианы считаются по всем объявлениям сегмента:

```
is_high_demand = 1, если одновременно:
    number_of_reviews_ltm > median(number_of_reviews_ltm по сегменту)
    price / median(price по сегменту) >= 1
    review_scores_rating / median(review_scores_rating по сегменту) >= 1
иначе 0
```

В коде колонка называется `numeric_of_reviews_ltm`. Доля положительного класса — **12.3%** (41 426 из 336 249).

Важно для интерпретации: таргет — детерминированное правило. Рейтинг, город и тип жилья остаются признаками модели, поэтому одно из трёх условий модель может восстановить напрямую (подробнее — раздел 12 и `docs/TECHNICAL_AUDIT.md`, R1).

## 3. Данные

PostgreSQL, схема `final_project`, по 10 городов (Amsterdam, Chicago, LA, London, Lyon, Munich, NYC, Paris, Toronto, Vienna):

| Таблица | Объём | Использование |
|---|---|---|
| `*_listings` | 661 425 строк, 65 колонок | Основа датасета |
| `*_calendars` | ~120 млн строк | Только EDA на выборке 5 000 строк из каждого города (колонка `price` в выборке полностью пустая) |
| `*_reviews` | ~10.6 млн строк | Только EDA текстов; в признаки не вошли |
| `SmartScore_test_snapshot` | — | Вход inference DAG (как таблица была сформирована — в репозитории не зафиксировано) |
| `SmartScore_predict` | — | Выход DAG |

Одна строка обучающей выборки — одно объявление (`source == "city scrape"`).

## 4. EDA и предобработка

`jupyter/EDA.ipynb`:
1. Объединение 10 таблиц listings с колонкой `city`; анализ пропусков, дубликатов `id` (297 341 повтор среди 661 425 строк), выбросов (`maximum_nights` до 2.1·10⁹, `price` в формате `"$90.00"`).
2. Оставлены записи `city scrape` (397 638), `price` приведена к числу, строки без цены удалены (→ 336 249), т.к. без цены таргет не определён.
3. Удалены колонки с утечкой: `price` и всё, что описывает реализованный спрос (`number_of_reviews*`, `reviews_per_month`, `first/last_review`, `estimated_occupancy/revenue_l365d`, `availability_*`), и вспомогательные колонки таргета (`median_*`, `*_ratio`).
4. Удалены идентификаторы и высококардинальные списки: `id`, `host_id`, `source`, `amenities`, `host_verifications`.
5. После сплита пропуски заполнены по train: числовые — медианой, категориальные — `"Unknown"`. Колонки с >50% пропусков (`neighbourhood_group_cleansed`, `neighborhood_overview`) в модель не вошли.

## 5. Feature engineering

Реализовано в `EDA.ipynb` (разделы 7–8) и повторено для инференса в `dags/preprocess.py::feature_engineering`:

| Признак | Формула |
|---|---|
| `avg_stay_nights` | `(minimum_nights + min(maximum_nights, 1125)) / 2` |
| `stay_ratio` | `minimum_nights / min(maximum_nights, 1125)` |
| `name_length`, `name_word_count`, `desc_length` | Длина/число слов текста **после** заполнения пропусков `"Unknown"` |
| `bathrooms_per_guest`, `beds_per_guest` | Отношение к `accommodates` (0 → NaN) |
| `text_features` | `name + " " + description`, Unicode NFKC |

Итог: 51 признак — 34 числовых и 17 категориальных. `name`, `description` и `text_features` переданы в CatBoost как **категориальные** (не как text features).

## 6. Разделение данных и валидация

* `train_test_split(test_size=0.2, stratify=y, random_state=42)` → train 268 999 / test 67 250.
* Внутри train: `train_test_split(test_size=0.2, stratify=y_train, random_state=42)` → train_fit 215 199 / valid 53 800.
* Сравнение моделей и подбор гиперпараметров — на valid; test использован один раз для финальной модели; порог классификации — 0.5 (по умолчанию, на тесте не подбирался).
* Ограничения: нет дедупликации по `id` перед случайным сплитом, нет временной валидации и кросс-валидации (см. раздел 12).

## 7. Модели и метрики

Основная метрика — **PR-AUC (average precision)**: класс редкий (12.3%), а для ранжирования важно качество в верхней части списка, которое ROC-AUC при дисбалансе показывает завышенно. Дополнительно — ROC-AUC, F1, Precision, Recall при пороге 0.5.

Дисбаланс учтён весами классов: `class_weight="balanced"` (LogReg, LightGBM), `scale_pos_weight=8` (XGBoost), `auto_class_weights="Balanced"` (CatBoost).

| Модель | Признаки | Valid ROC-AUC | Valid PR-AUC | Статус цифр |
|---|---|---:|---:|---|
| Logistic Regression | OHE + StandardScaler + TF-IDF (56 988 признаков) | 0.9241 | 0.6196 | Вывод ячейки 10 |
| CatBoost (500 it., depth 6) | 51 признак, нативные категории | 0.9727 | 0.8381 | Вывод ячейки 16 |
| XGBoost | Как у LogReg | 0.9655 | 0.7998 | Вывод ячейки 18 |
| LightGBM | Как у LogReg | — | — | Обучение прервано (`KeyboardInterrupt`), цифр нет |

Значения CatBoost `0.9919 / 0.9378` и LightGBM `0.9684` из таблицы сравнения в ноутбуке — устаревший вывод, кодом ноутбука они не воспроизводятся (`docs/TECHNICAL_AUDIT.md`, раздел 2).

**Optuna:** 15 trials TPE, цель — PR-AUC на valid, пространство: `iterations` 300–1000, `learning_rate` 0.01–0.1 (log), `depth` 4–10, `l2_leaf_reg` 1–10, `random_strength` 0–5, `bagging_temperature` 0–5. Лучший valid PR-AUC — 0.8558.

## 8. Финальная модель и результаты

`CatBoostClassifier` с параметрами из Optuna. Параметры проверены по сохранённому артефакту (`get_all_params()`): `iterations=752`, `learning_rate≈0.0997`, `depth=9`, `l2_leaf_reg≈9.25`, `random_strength≈0.362`, `auto_class_weights=Balanced`, `random_seed=42`. Обучена на train_fit (215 199 строк).

**Что известно о качестве именно сохранённой модели:**

| Выборка | Строк | Метрика | Источник | Статус |
|---|---:|---|---|---|
| Validation | 53 800 | PR-AUC **0.8558** | Лучший trial Optuna (ячейка 22) с теми же параметрами, данными и `random_seed`, что у сохранённой модели | Наиболее надёжная цифра; оптимистична, т.к. по этому же valid выбирались параметры |
| Test | 67 250 | — | — | **Не измерено** для сохранённой модели, нужно перепроверить по исходным экспериментам |

**Таблица из ноутбука, которая к сохранённой модели, по-видимому, не относится** (ячейка 33, `exec=None`):

| Выборка | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Validation | 0.9936 | 0.9527 | 0.8709 | 0.7969 | 0.9599 |
| Test | 0.9938 | 0.9548 | 0.8704 | 0.7939 | 0.9631 |

Почему не относится: переобучение с теми же параметрами на тех же данных не может поднять valid PR-AUC с 0.8558 до 0.9527. Вывод ячейки 33 сохранился с той же сессии, что и SHAP-анализ, где у модели было 56 признаков, включая `price` и `log_price`. `price` — компонент таргета (`price_ratio`), поэтому эта версия содержала утечку. Переоценить сохранённую модель в рамках аудита нельзя: `X_test.pkl` текущей версии нет в репозитории.

Важность признаков сохранённой модели (`get_feature_importance()`): `review_scores_rating` 18.4, `longitude` 9.3, `latitude` 6.2, `review_scores_value` 5.5, `room_type` 5.1, `neighbourhood_cleansed` 3.6. SHAP-таблица в ноутбуке построена для другой версии модели (56 признаков, с `price`) и к артефакту не относится.

## 9. Архитектура inference pipeline

```
            ┌──────────────────────┐   ┌───────────────────┐
            │ check_s3_artifacts   │   │ validate_input    │
            │ head_object ×3,      │   │ COUNT(*), схема,  │
            │ ETag модели → XCom   │   │ MAX(snapshot_date)│
            └──────────┬───────────┘   └────────┬──────────┘
                       └──────────┬─────────────┘
                                  ▼
                       ┌──────────────────────┐
                       │ predict              │
                       │ SELECT * snapshot    │
                       │ загрузка артефактов  │
                       │ проверка контракта   │
                       │ fill → FE → 51 приз. │
                       │ predict_proba        │
                       └──────────┬───────────┘
                                  ▼ XCom: {listing_id: [...], score: [...]}
                       ┌──────────────────────┐
                       │ save_predictions     │
                       │ DELETE за дату +     │
                       │ INSERT, 1 транзакция │
                       └──────────────────────┘
```

| Задача | Что делает | Код |
|---|---|---|
| `check_s3_artifacts` | Проверяет наличие трёх артефактов в S3, логирует ETag/дату/размер модели | `s3_utils.check_s3_objects` |
| `validate_input` | Падает на пустом snapshot или если нет одной из 43 нужных колонок; определяет `inference_date = MAX(snapshot_date)` | `postgres_utils.load_snapshot_metadata`, `get_inference_date` |
| `predict` | Читает snapshot, загружает артефакты, сверяет признаки модели с контрактом в коде, выполняет предобработку и `predict_proba` | `preprocess.preprocess_for_inference`, `inference.run_prediction` |
| `save_predictions` | Перезаписывает результаты за `inference_date` в одной транзакции | `save_results.save_predictions` |

Параметры DAG: `schedule=None` (запуск вручную, т.к. snapshot фиксированный), `max_active_runs=1`, `retries=2` с паузой 5 минут, `execution_timeout=1h`.

**Повторный запуск** за ту же дату перезаписывает результат (`DELETE` + `INSERT` в одной транзакции). При ошибке транзакция откатывается, и в таблице остаётся предыдущий полный результат.

## 10. Роль PostgreSQL, S3 и Airflow

* **PostgreSQL** — источник сырых данных (`final_project.*_listings`, `SmartScore_test_snapshot`) и хранилище результата `final_project.SmartScore_predict (listing_id BIGINT, score FLOAT, inference_date DATE)`. Имена без кавычек PostgreSQL хранит в нижнем регистре: фактически `smartscore_predict`. Connection: `postgres_sales_db`.
* **S3 (объектное хранилище)** — артефакты, которые DAG реально использует:
  * `final_catboost_model.pkl` — модель (joblib, ~211 МБ);
  * `fill_values.pkl` — правила заполнения пропусков;
  * `cat_features.pkl` — список категориальных признаков (сверяется с моделью; при расхождении используется список из модели).

  `preprocessor.pkl` (ColumnTransformer для LogReg/XGBoost) DAG **не использует**. Connection: `aws_default` (`login`/`password` — ключи, `extra.endpoint_url`, `extra.region_name`).
* **Airflow** — оркестрация, retries, логирование. Variables: `S3_BUCKET` (обязательная), `MODEL_KEY` (по умолчанию `models/final_catboost_model.pkl`), `FILL_VALUES_KEY`, `CAT_FEATURES_KEY`.

## 11. Структура проекта и запуск

```text
.
├── dags/
│   ├── smartscore_inference_dag.py  # DAG: 4 задачи
│   ├── preprocess.py                # контракт признаков, проверка схемы, fill → FE
│   ├── inference.py                 # сверка артефактов, predict_proba
│   ├── postgres_utils.py            # чтение snapshot и метаданных
│   ├── s3_utils.py                  # проверка и загрузка артефактов
│   └── save_results.py              # транзакционная запись результатов
├── jupyter/
│   ├── EDA.ipynb                    # данные, таргет, утечки, сплит, FE
│   └── Models.ipynb                 # эксперименты, Optuna, финальная модель, SHAP
├── scripts/rebuild_fill_values.py   # восстановление медиан из X_train.pkl
├── tests/                           # pytest: FE, схема, инференс, сохранение, документация
├── docs/                            # аудит, гайд к интервью, план развития
├── models/                          # локальные артефакты (в .gitignore)
├── requirements.txt / requirements-dev.txt
├── .env.example                     # шаблон jupyter/connect.env без значений
├── ds_base_final_project.md         # рабочий документ проекта
└── dag.md
```

**Тесты** (Airflow, БД и S3 не нужны; тесты с реальной моделью пропускаются, если нет `models/final_catboost_model.pkl`):

```bash
pip install -r requirements-dev.txt
```

```bash
python -m pytest -v
```

**Развёртывание DAG:** скопировать содержимое `dags/` в папку DAG-ов Airflow, установить `requirements.txt`, создать Connections `postgres_sales_db` и `aws_default`, Variable `S3_BUCKET`, загрузить три артефакта в S3 и запустить DAG `smartscore_inference` вручную.

**Секреты:** только в Airflow Connections и в локальном `jupyter/connect.env` (в `.gitignore`). Шаблон — `.env.example`.

## 12. Ограничения и нерешённые проблемы

* **Утечка учётных данных в истории git** (`connect.env` в коммите `3ef6a3a`). Нужны ротация ключей и очистка истории — действия владельца репозитория.
* **Таргет — правило от признаков того же снимка.** Высокие метрики отчасти отражают восстановление правила (рейтинг относительно медианы сегмента), а не предсказание будущего спроса. Временного разделения нет.
* **Дубликаты `id`** не удалены перед случайным сплитом (≥33 554 повторов в `city scrape`). Объявление может попасть и в train, и в test, а `name`/`description` как категории усиливают запоминание.
* **Медианы для числовых пропусков потеряны** (`fill_values.pkl` → `numeric: {}`). На инференсе NaN уходят в CatBoost как есть. Исправляется скриптом `scripts/rebuild_fill_values.py` при наличии `X_train.pkl`.
* **Ноутбуки выполнены не по порядку** и не воспроизводятся через Restart & Run All. Часть цифр в markdown не совпадает с выводами кода.
* **Модель структурно занижает новые объявления** (главный признак — рейтинг), хотя постановка просит обратного.
* **Нет мониторинга** дрейфа и распределения скоров; версия модели не пишется в таблицу результатов.
* DAG не запускался в Airflow в рамках аудита: Airflow и доступ к БД/S3 отсутствовали. Проверены unit-тесты и smoke-инференс реальной модели на синтетических данных.

**Влияние исправлений на предсказания.** Исправление train/serve skew меняет скоры: теперь модель получает признаки так же, как на обучении. На 2 000 синтетических строк старый и новый пайплайн совпали по Spearman на 0.87, топ-200 пересекается на 92.5%, при пороге 0.5 класс поменялся у 0.4% строк, максимальное расхождение скора — 0.41. Это оценка масштаба на синтетике, а не на реальных данных.
