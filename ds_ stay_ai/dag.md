# SmartScore Inference DAG

## Назначение

DAG выполняет пакетный инференс модели SmartScore. Он загружает тестовый срез данных из PostgreSQL, получает обученную модель из S3, выполняет предобработку данных, рассчитывает прогнозы и сохраняет результаты обратно в PostgreSQL.

---

## Схема работы DAG

```
Проверка модели в S3
        ↓
Загрузка тестового среза из PostgreSQL
        ↓
Определение даты инференса
        ↓
Загрузка модели и preprocessing из S3
        ↓
Предобработка данных
        ↓
Batch inference
        ↓
Сохранение результатов в PostgreSQL
```

---

## Описание задач

### check_s3

Проверяет доступность модели и preprocessing в S3.

---

### load_data

Загружает данные из таблицы

```
final_project.SmartScore_test_snapshot
```

---

### get_inference_date

Определяет дату инференса как максимальную дату в таблице SmartScore_test_snapshot.

---

### load_model

Загружает из S3:

- final_catboost_model.pkl
- preprocessor.pkl

---

### preprocess

Выполняет подготовку данных:

- обработку пропусков;
- генерацию новых признаков;
- применение сохранённого preprocessing pipeline.

---

### predict

Выполняет batch-инференс модели и рассчитывает вероятность высокого спроса для каждого объявления.

---

### save_results

Создаёт таблицу (если она отсутствует)

```
final_project.SmartScore_predict
```

и сохраняет результаты инференса.

При повторном запуске данные за уже существующую дату не записываются повторно.

---

# Источники данных

PostgreSQL

Основная таблица:

```
final_project.SmartScore_test_snapshot
```

Используется как фиксированный тестовый срез для выполнения инференса.

---

# Хранение артефактов

S3 Bucket

Хранятся:

- final_catboost_model.pkl
- preprocessor.pkl

Модель и preprocessing загружаются из S3 перед выполнением инференса.

---

# Результат работы

После успешного выполнения DAG в PostgreSQL создаётся (при необходимости) таблица

```
final_project.SmartScore_predict
```

которая содержит:

- listing_id
- score
- inference_date