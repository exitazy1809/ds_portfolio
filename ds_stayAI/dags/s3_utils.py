# s3_utils.py

import logging
import tempfile
import os
import pickle

import joblib


S3_CONN_ID = "aws_default"


# =====================================
# S3 client
# =====================================

def get_s3_client():
    """
    Клиент S3 из Airflow Connection aws_default.
    Импорты внутри функции, чтобы модуль
    можно было импортировать в тестах без Airflow.
    """

    import boto3

    try:
        from airflow.sdk import BaseHook
    except ImportError:
        from airflow.hooks.base import BaseHook

    conn = BaseHook.get_connection(
        S3_CONN_ID
    )

    return boto3.client(

        "s3",

        aws_access_key_id=conn.login,

        aws_secret_access_key=conn.password,

        endpoint_url=conn.extra_dejson.get(
            "endpoint_url"
        ),

        region_name=conn.extra_dejson.get(
            "region_name",
            "ru-central1"
        )

    )


# =====================================
# Проверка артефактов в S3
# =====================================

def check_s3_objects(
    bucket,
    keys,
    s3=None
):
    """
    Проверяет, что все артефакты есть в S3.
    Возвращает метаданные (ETag, дата, размер)
    для логирования версии модели.
    """

    s3 = s3 or get_s3_client()

    metadata = {}
    missing = []

    for key in keys:

        try:
            head = s3.head_object(
                Bucket=bucket,
                Key=key
            )
        except Exception as e:
            logging.error(
                "Объект недоступен %s/%s: %s",
                bucket,
                key,
                e
            )
            missing.append(key)
            continue

        metadata[key] = {
            "etag": str(head.get("ETag", "")).strip('"'),
            "last_modified": str(head.get("LastModified", "")),
            "size_bytes": int(head.get("ContentLength", 0)),
        }

        logging.info(
            "S3 %s/%s: %s",
            bucket,
            key,
            metadata[key]
        )

    if missing:
        raise FileNotFoundError(
            f"В S3 bucket {bucket} нет объектов: {missing}"
        )

    return metadata


# =====================================
# Load pickle from S3
# =====================================

def load_pickle_from_s3(
    bucket,
    key,
    s3=None
):

    logging.info(
        "Загрузка объекта из S3: %s/%s",
        bucket,
        key
    )

    s3 = s3 or get_s3_client()

    tmp_path = None

    try:

        response = s3.get_object(

            Bucket=bucket,

            Key=key

        )

        body = response["Body"].read()

        with tempfile.NamedTemporaryFile(

            suffix=".pkl",

            delete=False

        ) as tmp:

            tmp.write(body)

            tmp_path = tmp.name

        try:

            # сначала пробуем joblib

            obj = joblib.load(
                tmp_path
            )

        except Exception as joblib_error:

            # fallback pickle

            logging.warning(
                "joblib.load не смог прочитать %s (%s), пробуем pickle",
                key,
                joblib_error
            )

            with open(
                tmp_path,
                "rb"
            ) as f:

                obj = pickle.load(f)

        logging.info(
            "Объект загружен: %s",
            type(obj)
        )

        return obj

    except Exception as e:

        logging.error(
            "Ошибка загрузки %s/%s: %s",
            bucket,
            key,
            e
        )

        raise

    finally:

        if tmp_path and os.path.exists(tmp_path):

            os.remove(tmp_path)
