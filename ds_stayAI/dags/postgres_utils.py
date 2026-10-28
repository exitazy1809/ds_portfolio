# postgres_utils.py

import logging
import pandas as pd


POSTGRES_CONN_ID = "postgres_sales_db"

SNAPSHOT_TABLE = "final_project.SmartScore_test_snapshot"


def get_postgres_hook():
    """
    Создание подключения к PostgreSQL.
    Импорт внутри функции, чтобы модуль
    можно было импортировать в тестах без Airflow.
    """

    from airflow.providers.postgres.hooks.postgres import PostgresHook

    return PostgresHook(
        postgres_conn_id=POSTGRES_CONN_ID
    )


def load_test_snapshot():
    """
    Загружает тестовый срез данных
    для инференса.
    """

    hook = get_postgres_hook()

    sql = f"""
    SELECT *
    FROM {SNAPSHOT_TABLE}
    """

    df = hook.get_pandas_df(
        sql
    )

    if df.empty:
        raise ValueError(
            "Тестовый датасет пустой"
        )

    logging.info(
        "Загружено строк: %s",
        len(df)
    )

    return df


def load_snapshot_metadata():
    """
    Метаданные snapshot без выгрузки самих данных:
    число строк и список колонок.
    """

    hook = get_postgres_hook()

    row_count = hook.get_first(
        f"SELECT COUNT(*) FROM {SNAPSHOT_TABLE}"
    )[0]

    columns_df = hook.get_pandas_df(
        f"SELECT * FROM {SNAPSHOT_TABLE} LIMIT 0"
    )

    logging.info(
        "Snapshot: %s строк, %s колонок",
        row_count,
        len(columns_df.columns)
    )

    return int(row_count), list(columns_df.columns)


def get_inference_date():
    """
    Определяет дату инференса.
    Берём максимальную дату из snapshot.
    """

    hook = get_postgres_hook()

    sql = f"""
    SELECT MAX(snapshot_date)
    FROM {SNAPSHOT_TABLE}
    """

    result = hook.get_first(
        sql
    )

    if not result or result[0] is None:
        raise ValueError(
            "Не удалось определить дату инференса"
        )

    date = pd.Timestamp(
        result[0]
    )

    logging.info(
        "Дата инференса: %s",
        date
    )

    return date
