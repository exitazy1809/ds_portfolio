# save_results.py

import logging

import pandas as pd

from airflow.providers.postgres.hooks.postgres import PostgresHook


POSTGRES_CONN_ID = (
    "postgres_sales_db"
)


SCHEMA_NAME = (
    "final_project"
)


TABLE_NAME = (
    "SmartScore_predict"
)


def save_predictions(
    predictions,
    inference_date
):
    """
    Сохранение результатов инференса в Postgres.
    """

    logging.info(
        "Старт сохранения predictions"
    )

    # =====================================
    # Проверка колонок
    # =====================================

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

    predictions = predictions.copy()

    # =====================================
    # Приведение типов
    # Для совместимости psycopg2
    # =====================================

    predictions["listing_id"] = (

        pd.to_numeric(
            predictions["listing_id"],
            errors="coerce"
        )

    )

    predictions["score"] = (

        pd.to_numeric(
            predictions["score"],
            errors="coerce"
        )

    )

    if predictions["listing_id"].isna().any():

        raise ValueError(
            "Есть пустые listing_id"
        )

    if predictions["score"].isna().any():

        raise ValueError(
            "Есть пустые score"
        )

    # обычные Python типы,
    # не numpy.int64

    predictions["listing_id"] = (

        predictions["listing_id"]
        .astype(int)

    )

    predictions["score"] = (

        predictions["score"]
        .astype(float)

    )

    # =====================================
    # Дата инференса
    # =====================================

    predictions["inference_date"] = (

        inference_date.date()

    )

    logging.info(

        "Подготовлено строк: %s",

        len(predictions)

    )

    # =====================================
    # Подключение Postgres
    # =====================================

    hook = PostgresHook(

        postgres_conn_id=POSTGRES_CONN_ID

    )

    # =====================================
    # Создание таблицы
    # =====================================

    create_sql = f"""

    CREATE TABLE IF NOT EXISTS
    {SCHEMA_NAME}.{TABLE_NAME}
    (

        listing_id BIGINT,

        score FLOAT,

        inference_date DATE

    );

    """

    hook.run(

        create_sql

    )

    # =====================================
    # Удаляем прогноз за эту дату
    # =====================================

    delete_sql = f"""

    DELETE FROM
    {SCHEMA_NAME}.{TABLE_NAME}

    WHERE inference_date = %s;

    """

    hook.run(

        delete_sql,

        parameters=[

            inference_date.date()

        ]

    )

    # =====================================
    # Подготовка строк для psycopg2
    # Только Python int/float/date
    # =====================================

    rows = [

        (

            int(row.listing_id),

            float(row.score),

            row.inference_date

        )

        for row in predictions[

            [

                "listing_id",

                "score",

                "inference_date"

            ]

        ].itertuples(

            index=False

        )

    ]

    logging.info(

        "Пример строки вставки: %s",

        rows[0] if rows else None

    )

    logging.info(

        "Типы данных строки: %s",

        [

            type(x)

            for x in rows[0]

        ]

        if rows

        else None

    )

    # =====================================
    # Insert
    # =====================================

    hook.insert_rows(

        table=f"{SCHEMA_NAME}.{TABLE_NAME}",

        rows=rows,

        target_fields=[

            "listing_id",

            "score",

            "inference_date"

        ]

    )

    logging.info(

        "Сохранено строк: %s",

        len(rows)

    )

    logging.info(

        "Таблица назначения: %s.%s",

        SCHEMA_NAME,

        TABLE_NAME

    )
