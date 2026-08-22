# postgres_utils.py

import logging
import pandas as pd

from airflow.providers.postgres.hooks.postgres import PostgresHook


POSTGRES_CONN_ID = "postgres_sales_db"


def get_postgres_hook():
    """
    Создание подключения к PostgreSQL.
    """

    return PostgresHook(
        postgres_conn_id=POSTGRES_CONN_ID
    )


def load_test_snapshot():
    """
    Загружает тестовый срез данных
    для инференса.
    """

    hook = get_postgres_hook()

    sql = """
    SELECT *
    FROM final_project.SmartScore_test_snapshot
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


def get_inference_date():
    """
    Определяет дату инференса.
    Берём максимальную дату из snapshot.
    """

    hook = get_postgres_hook()

    sql = """
    SELECT MAX(snapshot_date)
    FROM final_project.SmartScore_test_snapshot
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
