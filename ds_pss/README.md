# Пайплайн прогнозирования продаж

## База данных

DB: playground_ds_20260415_fef6b35a86  
User: ds_20260415_fef6b35a86

## Бизнес-задача

Автоматизация прогнозирования недельных продаж (`weekly_sales`) для магазинов и отделов:

- Планирование продаж
- Прогнозирование нагрузки
- Аналитика магазинов
- Поддержка бизнес-решений

## Архитектура проекта

- dags/dag.py Основной Airflow DAG
- dags/preprocessing.py # Логика предобработки признаков

- model_training.ipynb # Обучение и оценка модели
- requirements.txt # Зависимости
- README.md


## Используемые технологии

- Оркестрация: Apache Airflow 
- ML-фреймворк: CatBoost 
- База данных: PostgreSQL 
- Хранилище: S3
- Обработка данных :Pandas, NumPy 
- SDK: boto3 

## Источники данных

Таблица

- sales: Исторические продажи
- plan: Периоды предсказания 
- stores: Информация о магазинах
- features: Внешние признаки 

## Предобработка признаков

Модуль: dags/preprocessing.py

### Генерируемые признаки

Категориальные:
- store
- dept
- type
- is_holiday

Временные:
- month
- quarter
- year
- week

Лаговые признаки:
- sales_1week_ago
- sales_2week_ago
- sales_4week_ago

Rolling статистики:
- mean_sales_2week
- mean_sales_4week
- avg_sales_before

### Этапы предобработки

- обработка аномалий
- заполнение пропусков
- генерация временных признаков
- создание лагов
- rolling статистики
- удаление NaN после feature engineering

## Модель

- алгоритм: CatBoostRegressor
- метрики: RMSE, MAE, R²
- хранение: Yandex Object Storage (S3)
- обучение: model_training.ipynb

## Архитектура DAG

1. sales_prediction_batch_inference(DAG)

2. load_data_from_postgres
    
3. preprocess_features
    
4. load_model_from_s3
    
5. run_batch_inference
    
6. save_predictions_to_postgres

## Инструкция по запуску

### Установка зависимостей

pip install -r requirements.txt

### Airflow Connections

Connection ID: postgres_sales_db

Host: <host>  
Port: <port>  
Database: playground_ds_20260415_fef6b35a86  
Login: ds_20260415_fef6b35a86  
Password: <password>

### Airflow Variables

- <s3_access_key>
- <s3_secret_key>
- <s3_bucket_name>
- <s3_model_key>

### Запуск DAG

- открыть Airflow UI
- выбрать sales_prediction_batch_inference
- Trigger DAG
- мониторинг в Graph View

## Проверка успешного выполнения

DAG считается успешным если:

- все задачи выполнены (success)
- таблица predictions заполнена в PostgreSQL
- в логах есть:
  - количество строк
  - диапазон дат
  - статистика предсказаний

## Безопасность и production-подход

- секреты через Airflow Connections и Variables
- отсутствие хардкода
- XCom только для малого объема
- единая логика train/inference
- контроль признаков и утечки
- обработка отрицательных предсказаний

## Зависимости

apache-airflow  
pandas  
numpy  
catboost  
boto3  
psycopg2-binary  
scikit-learn  

## Хранилище

https://storage.yandexcloud.net