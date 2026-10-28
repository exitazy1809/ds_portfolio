# SmartScore Inference DAG

## Назначение

DAG `smartscore_inference` (`dags/smartscore_inference_dag.py`) выполняет пакетный инференс модели SmartScore. Он проверяет артефакты в S3 и входной срез в PostgreSQL, рассчитывает вероятность высокого спроса для каждого объявления и сохраняет результаты обратно в PostgreSQL.

---

## Схема

```
check_s3_artifacts ──┐
                     ├──> predict ──> save_predictions
validate_input ──────┘
```

`check_s3_artifacts` и `validate_input` независимы и могут выполняться параллельно.

---

## Параметры DAG

| Параметр | Значение | Зачем |
|---|---|---|
| `schedule` | `None` | Запуск вручную: входной snapshot фиксированный |
| `catchup` | `False` | |
| `max_active_runs` | `1` | Два запуска не перемешивают `DELETE`/`INSERT` |
| `retries` | `2`, пауза 5 минут | Переживает временные сбои БД/S3 |
| `execution_timeout` | 1 час | |

---

## Задачи

### `check_s3_artifacts`

Проверяет через `head_object`, что в бакете `S3_BUCKET` есть все три артефакта:

- `final_catboost_model.pkl` (`MODEL_KEY`)
- `fill_values.pkl` (`FILL_VALUES_KEY`)
- `cat_features.pkl` (`CAT_FEATURES_KEY`)

Если хотя бы одного нет, задача падает со списком отсутствующих ключей. ETag, дата изменения и размер модели передаются в XCom (`model_version`) и пишутся в лог задачи `predict`.

### `validate_input`

- считает строки в `final_project.SmartScore_test_snapshot` и падает, если таблица пустая;
- проверяет, что в snapshot есть все 43 сырые колонки, нужные модели (`preprocess.RAW_REQUIRED_COLUMNS`);
- определяет дату инференса как `MAX(snapshot_date)`.

В XCom (`input_meta`) передаются только `row_count` и `inference_date`. Сам DataFrame через XCom не передаётся: JSON-сериализация портит даты, булевы значения и длинные `listing_id`.

### `predict`

1. Читает snapshot из PostgreSQL.
2. Загружает артефакты из S3.
3. Сверяет `model.feature_names_` с контрактом в коде (`preprocess.MODEL_FEATURES`, 51 признак). При расхождении задача падает. Категориальные признаки берутся из модели (`get_cat_feature_indices`); если `cat_features.pkl` с ними не совпадает, пишется warning.
4. `preprocess_for_inference`: проверка схемы → удаление `is_high_demand`, если колонка есть → заполнение пропусков по `fill_values.pkl` → feature engineering → выбор 51 признака в порядке модели.
5. `predict_proba`, в XCom (`prediction`) кладутся два списка: `listing_id` и `score`.

### `save_predictions`

В **одной транзакции**:

1. `CREATE TABLE IF NOT EXISTS final_project.SmartScore_predict (listing_id BIGINT, score FLOAT, inference_date DATE)`;
2. `DELETE` строк за `inference_date`;
3. `INSERT` всех предсказаний (`execute_values`).

При ошибке выполняется `rollback`, и в таблице остаётся предыдущий полный результат.

---

## Повторный запуск

Повторный запуск за ту же `inference_date` **перезаписывает** результаты этой даты: дубликатов не появляется, а при ошибке старые данные сохраняются.

---

## Конфигурация Airflow

| Тип | Имя | Назначение |
|---|---|---|
| Connection | `postgres_sales_db` | PostgreSQL с данными и результатами |
| Connection | `aws_default` | S3: `login`/`password` — ключи, `extra`: `endpoint_url`, `region_name` |
| Variable | `S3_BUCKET` | Обязательная |
| Variable | `MODEL_KEY` | По умолчанию `models/final_catboost_model.pkl` |
| Variable | `FILL_VALUES_KEY` | По умолчанию `models/fill_values.pkl` |
| Variable | `CAT_FEATURES_KEY` | По умолчанию `models/cat_features.pkl` |

Variables читаются внутри задач, а не при парсинге DAG.

---

## Результат

Таблица `final_project.SmartScore_predict` (в PostgreSQL хранится как `smartscore_predict`):

- `listing_id` — идентификатор объявления (`listing_id` или `id` из snapshot);
- `score` — вероятность высокого спроса, [0, 1];
- `inference_date` — `MAX(snapshot_date)` входного среза.
