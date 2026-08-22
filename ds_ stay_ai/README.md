 # SmartScore Inference Pipeline

## Описание проекта

Проект реализует пакетный инференс модели машинного обучения для прогнозирования высокого спроса на объекты аренды.

Инференс выполняется с помощью Apache Airflow.

Основные этапы:

- загрузка тестового среза из PostgreSQL;
- определение даты инференса;
- загрузка модели и preprocessing из S3;
- подготовка данных;
- batch-инференс;
- сохранение результатов в PostgreSQL.



БД: ds_20260415_fef6b35a86
---



## Структура проекта

```text
.
├── dags/
│   ├── smartscore_inference_dag.py
│   ├── preprocess.py
│   ├── postgres_utils.py
│   ├── s3_utils.py
│   ├── inference.py
│   └── save_results.py
│
├── requirements.txt
├── dag.md
└── README.md
```

---

## Используемые технологии

- Python
- Apache Airflow
- PostgreSQL
- S3 (Object Storage)
- LightGBM
- pandas
- NumPy
- scikit-learn
- boto3

---

## Источники данных

### PostgreSQL

Тестовый срез:

```
final_project.SmartScore_test_snapshot
```

Результаты инференса:

```
final_project.SmartScore_predict
```

### S3

Хранятся артефакты модели:

- final_catboost_model.pkl
- preprocessor.pkl

---

## Порядок работы DAG

```
check_s3
      ↓
load_data
      ↓
get_inference_date
      ↓
load_model
      ↓
preprocess
      ↓
predict
      ↓
save_results
```

---

## Установка зависимостей

```bash
pip install -r requirements.txt
```

---

## Основные модули

| Файл | Назначение |
|------|------------|
| smartscore_inference_dag.py | описание DAG |
| preprocess.py | подготовка данных |
| postgres_utils.py | работа с PostgreSQL |
| s3_utils.py | загрузка модели из S3 |
| inference.py | выполнение инференса |
| save_results.py | сохранение результатов |

---

## Результат работы

После успешного выполнения DAG в PostgreSQL создаётся или обновляется таблица

```
final_project.SmartScore_predict
```

с полями:

- listing_id
- score
- inference_date

---
