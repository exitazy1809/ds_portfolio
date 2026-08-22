# smartscore.py

import logging

from datetime import datetime

import pandas as pd


from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable


from postgres_utils import (
    load_test_snapshot,
    get_inference_date
)


from s3_utils import (
    load_pickle_from_s3
)


from preprocess import (
    preprocess_for_inference
)


from inference import (
    run_prediction
)


from save_results import (
    save_predictions
)


# =====================================
# Airflow Variables
# =====================================

S3_BUCKET = Variable.get(
    "S3_BUCKET"
)


MODEL_KEY = Variable.get(
    "MODEL_KEY"
)


FILL_VALUES_KEY = Variable.get(
    "FILL_VALUES_KEY",
    default_var="models/fill_values.pkl"
)


CAT_FEATURES_KEY = Variable.get(
    "CAT_FEATURES_KEY",
    default_var="models/cat_features.pkl"
)


# =====================================
# Load data
# =====================================

def load_data_task(**context):

    df = load_test_snapshot()

    if df.empty:

        raise ValueError(
            "Входная таблица пустая"
        )

    logging.info(
        "Загружено строк: %s",
        len(df)
    )

    logging.info(
        "Колонки: %s",
        df.columns.tolist()
    )

    context["ti"].xcom_push(

        key="data",

        value=df.to_json()

    )


# =====================================
# Inference date
# =====================================

def get_date_task(**context):

    date = get_inference_date()

    logging.info(
        "Дата инференса: %s",
        date
    )

    context["ti"].xcom_push(

        key="date",

        value=date.isoformat()

    )


# =====================================
# Prediction
# =====================================

def predict_task(**context):

    ti = context["ti"]

    # =================================
    # Получаем данные
    # =================================

    df_json = ti.xcom_pull(

        key="data",

        task_ids="load_test_data"

    )

    df = pd.read_json(
        df_json
    )

    logging.info(
        "Получено строк: %s",
        len(df)
    )

    # =================================
    # Получаем дату инференса
    # =================================

    inference_date = ti.xcom_pull(

        key="date",

        task_ids="get_inference_date"

    )

    inference_date = datetime.fromisoformat(

        inference_date

    )

    logging.info(

        "Дата для FE: %s",

        inference_date

    )

    # =================================
    # ID
    # =================================

    if "listing_id" in df.columns:

        listing_ids = df["listing_id"].copy()

    elif "id" in df.columns:

        listing_ids = df["id"].copy()

    else:

        raise ValueError(

            "В данных отсутствует listing_id или id"

        )

    # =================================
    # Model
    # =================================

    model = load_pickle_from_s3(

        bucket=S3_BUCKET,

        key=MODEL_KEY

    )

    if model is None:

        raise ValueError(
            "Модель не загружена"
        )

    logging.info(

        "Модель загружена: %s",

        type(model)

    )

    # =================================
    # Fill values
    # =================================

    fill_values = load_pickle_from_s3(

        bucket=S3_BUCKET,

        key=FILL_VALUES_KEY

    )

    logging.info(

        "Правила заполнения загружены"

    )

    # =================================
    # Cat features
    # =================================

    cat_features = load_pickle_from_s3(

        bucket=S3_BUCKET,

        key=CAT_FEATURES_KEY

    )

    logging.info(

        "Категориальных признаков: %s",

        len(cat_features)

    )

    # =================================
    # Preprocess
    # =================================

    X = preprocess_for_inference(

        df,

        inference_date

    )

    logging.info(

        "После preprocessing: %s",

        X.shape

    )

    # =================================
    # Prediction
    # =================================

    result = run_prediction(

        model,

        X,

        listing_ids,

        fill_values,

        cat_features

    )

    logging.info(

        "Получено результатов: %s",

        len(result)

    )

    ti.xcom_push(

        key="prediction",

        value=result.to_dict("records")

    )


# =====================================
# Save
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

    date = ti.xcom_pull(

        key="date",

        task_ids="get_inference_date"

    )

    date = datetime.fromisoformat(

        date

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
with DAG(

    dag_id="smartscore_inference",


    start_date=datetime(

        2026,

        1,

        1

    ),


    schedule=None,


    catchup=False,


    tags=[

        "smartscore",

        "ml",

        "catboost"

    ]

) as dag:

    load_data = PythonOperator(

        task_id="load_test_data",

        python_callable=load_data_task

    )

    get_date = PythonOperator(

        task_id="get_inference_date",

        python_callable=get_date_task

    )

    predict = PythonOperator(

        task_id="predict",

        python_callable=predict_task

    )

    save = PythonOperator(

        task_id="save_predictions",

        python_callable=save_task

    )

    # правильный порядок

    load_data >> predict

    get_date >> predict

    predict >> save
