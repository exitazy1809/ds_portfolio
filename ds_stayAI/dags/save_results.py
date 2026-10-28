# save_results.py

import logging

import pandas as pd


POSTGRES_CONN_ID = (
    "postgres_sales_db"
)


SCHEMA_NAME = (
    "final_project"
)


TABLE_NAME = (
    "SmartScore_predict"
)


CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS
{SCHEMA_NAME}.{TABLE_NAME}
(
    listing_id BIGINT,
    score FLOAT,
    inference_date DATE
);
"""

DELETE_SQL = f"""
DELETE FROM {SCHEMA_NAME}.{TABLE_NAME}
WHERE inference_date = %s;
"""

INSERT_SQL = f"""
INSERT INTO {SCHEMA_NAME}.{TABLE_NAME}
(listing_id, score, inference_date)
VALUES %s
"""


# =====================================
# Подготовка строк
# =====================================

def prepare_prediction_rows(
    predictions,
    inference_date
):
    """
    Проверяет predictions и превращает их
    в список кортежей (int, float, date) для psycopg2.
    """

    required_columns = [
        "listing_id",
        "score"
    ]

    missing = (
        set(required_columns)
        -
        set(predictions.columns)
    )

    if missing:
        raise ValueError(
            f"Отсутствуют обязательные колонки: {missing}"
        )

    if predictions.empty:
        raise ValueError(
            "Нет предсказаний для сохранения"
        )

    predictions = predictions.copy()

    predictions["listing_id"] = pd.to_numeric(
        predictions["listing_id"],
        errors="coerce"
    )

    predictions["score"] = pd.to_numeric(
        predictions["score"],
        errors="coerce"
    )

    if predictions["listing_id"].isna().any():
        raise ValueError(
            "Есть пустые listing_id"
        )

    if predictions["score"].isna().any():
        raise ValueError(
            "Есть пустые score"
        )

    if not predictions["score"].between(0, 1).all():
        raise ValueError(
            "score вне диапазона [0, 1]"
        )

    duplicated = predictions["listing_id"].duplicated().sum()

    if duplicated:
        logging.warning(
            "Повторяющихся listing_id: %s. "
            "Строки сохраняются как есть",
            duplicated
        )

    date = pd.Timestamp(inference_date).date()

    # обычные Python типы, не numpy.int64
    return [
        (
            int(listing_id),
            float(score),
            date
        )
        for listing_id, score in zip(
            predictions["listing_id"],
            predictions["score"]
        )
    ]


# =====================================
# Запись в одной транзакции
# =====================================

def write_prediction_rows(
    conn,
    rows,
    inference_date,
    page_size=5000
):
    """
    DELETE за дату + INSERT в одной транзакции.
    При ошибке — rollback: в таблице остаётся
    предыдущий полный результат, а не частичный.
    """

    from psycopg2.extras import execute_values

    date = pd.Timestamp(inference_date).date()

    try:

        with conn.cursor() as cur:

            cur.execute(CREATE_SQL)

            cur.execute(DELETE_SQL, (date,))

            logging.info(
                "Удалено старых строк за %s: %s",
                date,
                cur.rowcount
            )

            execute_values(
                cur,
                INSERT_SQL,
                rows,
                page_size=page_size
            )

        conn.commit()

    except Exception:

        conn.rollback()

        logging.error(
            "Ошибка записи, транзакция откатена"
        )

        raise


def save_predictions(
    predictions,
    inference_date
):
    """
    Сохранение результатов инференса в Postgres.
    Повторный запуск за ту же дату перезаписывает результат.
    """

    logging.info(
        "Старт сохранения predictions"
    )

    rows = prepare_prediction_rows(
        predictions,
        inference_date
    )

    logging.info(
        "Подготовлено строк: %s, пример: %s",
        len(rows),
        rows[0]
    )

    from airflow.providers.postgres.hooks.postgres import PostgresHook

    hook = PostgresHook(
        postgres_conn_id=POSTGRES_CONN_ID
    )

    conn = hook.get_conn()

    try:
        write_prediction_rows(
            conn,
            rows,
            inference_date
        )
    finally:
        conn.close()

    logging.info(
        "Сохранено строк: %s в %s.%s",
        len(rows),
        SCHEMA_NAME,
        TABLE_NAME
    )
