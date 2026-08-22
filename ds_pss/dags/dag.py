from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.hooks.postgres_hook import PostgresHook

from psycopg2.extras import execute_values

import pandas as pd
import numpy as np
import boto3
import io
import pickle
import logging

from preprocessing import preprocess_data

# ========== CONNECTION ==========
S3_BUCKET = Variable.get("s3_bucket_name")
S3_ACCESS_KEY = Variable.get("s3_access_key")
S3_SECRET_KEY = Variable.get("s3_secret_key")
S3_MODEL_KEY = Variable.get("s3_model_key")

POSTGRES_CONN_ID = "postgres_sales_db"

logger = logging.getLogger(__name__)


# ========== 1. LOAD DATA ==========
def load_data_from_postgres(**context):
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    query = """
    DROP TABLE IF EXISTS inference_data_temp;

    CREATE TABLE inference_data_temp AS
    WITH plan_data AS (
        SELECT store, dept, date,
               NULL::float AS weekly_sales,
               FALSE AS is_holiday
        FROM plan
    ),
    historical_data AS (
        SELECT store, dept, date,
               weekly_sales, is_holiday
        FROM sales
    ),
    combined AS (
        SELECT * FROM historical_data
        UNION ALL
        SELECT * FROM plan_data
    )
    SELECT
        c.store, c.dept, c.date,
        c.weekly_sales, c.is_holiday,
        st.type, st.size,
        f.temperature, f.fuel_price,
        f.factor1, f.factor2, f.factor3,
        f.factor4, f.factor5,
        f.cpi, f.unemployment
    FROM combined c
    LEFT JOIN stores st ON c.store = st.store
    LEFT JOIN features f
        ON c.store = f.store
       AND c.dept = f.dept
       AND c.date = f.date;
    """

    cursor.execute(query)
    conn.commit()

    cursor.execute("""
        SELECT COUNT(*), MIN(date), MAX(date)
        FROM inference_data_temp
    """)
    count, min_date, max_date = cursor.fetchone()

    logger.info(f"rows={count}, min={min_date}, max={max_date}")

    cursor.execute("SELECT MIN(date) FROM plan")
    first_plan_date = cursor.fetchone()[0]

    context['ti'].xcom_push(
        key='first_plan_date',
        value=str(first_plan_date)
    )

    cursor.close()
    conn.close()


# ========== 2. PREPROCESS ==========
def preprocess_features(**context):
    ti = context['ti']

    first_plan_date = pd.to_datetime(
        ti.xcom_pull(task_ids='load_data', key='first_plan_date')
    )

    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM inference_data_temp;")

    cols = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=cols)

    cursor.close()
    conn.close()

    df['date'] = pd.to_datetime(df['date'])

    df = preprocess_data(df)
    df = df[df['date'] >= first_plan_date]

    if 'weekly_sales' in df.columns:
        df = df.drop(columns=['weekly_sales'])

    df = df.dropna()

    ti.xcom_push(
        key='preprocessed_data',
        value=df.to_json(orient='records')
    )

# ========== 3. LOAD MODEL ==========
def load_model_from_s3(**context):
    s3 = boto3.client(
        's3',
        endpoint_url='https://storage.yandexcloud.net',
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name='ru-central1'
    )

    buffer = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, S3_MODEL_KEY, buffer)
    buffer.seek(0)

    model = pickle.load(buffer)

    path = "/tmp/model.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)

    context['ti'].xcom_push(key='model_path', value=path)


# ========== 4. INFERENCE ==========
def run_batch_inference(**context):
    ti = context['ti']

    df = pd.read_json(
        io.StringIO(
            ti.xcom_pull(task_ids='preprocess_features', key='preprocessed_data')
        ),
        orient='records'
    )

    model_path = ti.xcom_pull(task_ids='load_model', key='model_path')

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df['date'] = pd.to_datetime(df['date'])

    if 'week' not in df.columns:
        df['week'] = df['date'].dt.isocalendar().week.astype(int)

    expected_features = model.feature_names_

    df_model = df[expected_features]

    preds = model.predict(df_model)

    df['predicted_weekly_sales'] = np.maximum(preds, 0)

    ti.xcom_push(
        key='predictions',
        value=df.to_json(orient='records')
    )


# ========== 5. SAVE ==========
def save_predictions_to_postgres(**context):
    ti = context['ti']

    df = pd.read_json(
        io.StringIO(
            ti.xcom_pull(task_ids='run_inference', key='predictions')
        ),
        orient='records'
    )

    df['date'] = pd.to_datetime(df['date']).dt.date

    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

  
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        store INT,
        dept INT,
        date DATE,
        predicted_weekly_sales FLOAT,
        prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (store, dept, date)
    );
    """)
    conn.commit()

    values = df[['store', 'dept', 'date', 'predicted_weekly_sales']] \
        .to_records(index=False).tolist()

    execute_values(cursor, """
        INSERT INTO predictions (
            store, dept, date, predicted_weekly_sales
        )
        VALUES %s
        ON CONFLICT (store, dept, date)
        DO UPDATE SET
            predicted_weekly_sales = EXCLUDED.predicted_weekly_sales,
            prediction_timestamp = CURRENT_TIMESTAMP;
    """, values)

    conn.commit()

    cursor.execute("DROP TABLE IF EXISTS inference_data_temp;")
    conn.commit()

    cursor.close()
    conn.close()

    logger.info(f"saved rows: {len(df)}")


# ========== DAG ==========
default_args = {
    "owner": "daniil",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    "sales_prediction_batch_inference",
    default_args=default_args,
    schedule_interval="0 20 * * 0",
    start_date=datetime(2025, 1, 1),
    catchup=False
)


task_load_data = PythonOperator(
    task_id="load_data",
    python_callable=load_data_from_postgres,
    dag=dag,
)

task_preprocess = PythonOperator(
    task_id="preprocess_features",
    python_callable=preprocess_features,
    dag=dag,
)

task_load_model = PythonOperator(
    task_id="load_model",
    python_callable=load_model_from_s3,
    dag=dag,
)

task_inference = PythonOperator(
    task_id="run_inference",
    python_callable=run_batch_inference,
    dag=dag,
)

task_save = PythonOperator(
    task_id="save_predictions",
    python_callable=save_predictions_to_postgres,
    dag=dag,
)


task_load_data >> task_preprocess
task_preprocess >> task_inference
task_load_model >> task_inference
task_inference >> task_save
