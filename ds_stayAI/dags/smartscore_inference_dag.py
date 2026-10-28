# smartscore_inference_dag.py

import logging

from datetime import datetime, timedelta

import pandas as pd


from airflow import DAG

try:
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:
    from airflow.operators.python import PythonOperator

try:
    from airflow.sdk import Variable
except ImportError:
    from airflow.models import Variable


from postgres_utils import (
    load_test_snapshot,
    load_snapshot_metadata,
    get_inference_date
)


from s3_utils import (
    check_s3_objects,
    load_pickle_from_s3
)


from preprocess import (
    MODEL_FEATURES,
    RAW_REQUIRED_COLUMNS,
    SchemaError,
    get_listing_ids,
    preprocess_for_inference
)


from inference import (
    check_artifacts_consistency,
    run_prediction
)


from save_results import (
    save_predictions
)


# =====================================
# Airflow Variables
# Читаются внутри задач, а не при парсинге DAG
# =====================================

def get_config():

    # значение по умолчанию передаётся позиционно:
    # в Airflow 2 параметр называется default_var,
    # в airflow.sdk (Airflow 3) — default
    return {

        "bucket": Variable.get("S3_BUCKET"),

        "model_key": Variable.get(
            "MODEL_KEY",
            "models/final_catboost_model.pkl"
        ),

        "fill_values_key": Variable.get(
            "FILL_VALUES_KEY",
            "models/fill_values.pkl"
        ),

        "cat_features_key": Variable.get(
            "CAT_FEATURES_KEY",
            "models/cat_features.pkl"
        ),

    }


# =====================================
# 1. Проверка артефактов в S3
# =====================================

def check_s3_task(**context):

    config = get_config()

    metadata = check_s3_objects(

        bucket=config["bucket"],

        keys=[
            config["model_key"],
            config["fill_values_key"],
            config["cat_features_key"]
        ]

    )

    # версия модели для логов и последующих задач
    context["ti"].xcom_push(

        key="model_version",

        value={
            "key": config["model_key"],
            **metadata[config["model_key"]]
        }

    )


# =====================================
# 2. Проверка входных данных
# Через XCom передаются только метаданные,
# а не весь DataFrame
# =====================================

def validate_input_task(**context):

    row_count, columns = load_snapshot_metadata()

    if row_count == 0:
        raise ValueError(
            "Входная таблица пустая"
        )

    missing = [
        col for col in RAW_REQUIRED_COLUMNS
        if col not in columns
    ]

    if missing:
        raise SchemaError(
            f"В snapshot нет колонок, нужных модели: {missing}"
        )

    date = get_inference_date()

    logging.info(
        "Snapshot: %s строк, дата инференса %s",
        row_count,
        date
    )

    context["ti"].xcom_push(

        key="input_meta",

        value={
            "row_count": row_count,
            "inference_date": date.isoformat()
        }

    )


# =====================================
# 3. Prediction
# =====================================

def predict_task(**context):

    ti = context["ti"]

    config = get_config()

    input_meta = ti.xcom_pull(
        key="input_meta",
        task_ids="validate_input"
    )

    model_version = ti.xcom_pull(
        key="model_version",
        task_ids="check_s3_artifacts"
    )

    logging.info(
        "Модель: %s",
        model_version
    )

    # =================================
    # Данные читаются напрямую из PostgreSQL
    # =================================

    df = load_test_snapshot()

    if len(df) != input_meta["row_count"]:
        logging.warning(
            "Число строк изменилось после проверки: %s -> %s",
            input_meta["row_count"],
            len(df)
        )

    listing_ids = get_listing_ids(df)

    # =================================
    # Артефакты
    # =================================

    model = load_pickle_from_s3(
        bucket=config["bucket"],
        key=config["model_key"]
    )

    fill_values = load_pickle_from_s3(
        bucket=config["bucket"],
        key=config["fill_values_key"]
    )

    cat_features_artifact = load_pickle_from_s3(
        bucket=config["bucket"],
        key=config["cat_features_key"]
    )

    cat_features = check_artifacts_consistency(
        model,
        MODEL_FEATURES,
        cat_features_artifact
    )

    # =================================
    # Preprocess + prediction
    # =================================

    X = preprocess_for_inference(
        df,
        fill_values,
        model_features=MODEL_FEATURES,
        cat_features=cat_features
    )

    result = run_prediction(
        model,
        X,
        listing_ids
    )

    # компактный формат: два списка вместо списка словарей
    ti.xcom_push(

        key="prediction",

        value={
            "listing_id": [int(x) for x in result["listing_id"]],
            "score": [float(x) for x in result["score"]]
        }

    )


# =====================================
# 4. Save
# =====================================

def save_task(**context):

    ti = context["ti"]

    result_data = ti.xcom_pull(
        key="prediction",
        task_ids="predict"
    )

    result = pd.DataFrame(
        result_data
    )

    input_meta = ti.xcom_pull(
        key="input_meta",
        task_ids="validate_input"
    )

    date = datetime.fromisoformat(
        input_meta["inference_date"]
    )

    save_predictions(
        result,
        date
    )

    logging.info(
        "Результаты сохранены: %s строк",
        len(result)
    )


# =====================================
# DAG
# =====================================

default_args = {

    "owner": "smartscore",

    "retries": 2,

    "retry_delay": timedelta(minutes=5),

    "execution_timeout": timedelta(hours=1),

}


with DAG(

    dag_id="smartscore_inference",

    start_date=datetime(2026, 1, 1),

    # запуск вручную: snapshot фиксированный
    schedule=None,

    catchup=False,

    # повторные запуски не пересекаются
    max_active_runs=1,

    default_args=default_args,

    tags=[
        "smartscore",
        "ml",
        "catboost"
    ]

) as dag:

    check_s3 = PythonOperator(
        task_id="check_s3_artifacts",
        python_callable=check_s3_task
    )

    validate_input = PythonOperator(
        task_id="validate_input",
        python_callable=validate_input_task
    )

    predict = PythonOperator(
        task_id="predict",
        python_callable=predict_task
    )

    save = PythonOperator(
        task_id="save_predictions",
        python_callable=save_task
    )

    [check_s3, validate_input] >> predict >> save
