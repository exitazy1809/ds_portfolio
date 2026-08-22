# s3_utils.py

import logging
import tempfile
import os
import pickle

import boto3
import joblib

from airflow.hooks.base import BaseHook


# =====================================
# S3 client
# =====================================

def get_s3_client():

    conn = BaseHook.get_connection(
        "aws_default"
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
# Load pickle from S3
# =====================================

def load_pickle_from_s3(
    bucket,
    key
):

    logging.info(
        "Загрузка объекта из S3: %s/%s",
        bucket,
        key
    )

    s3 = get_s3_client()

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

        except Exception:

            # fallback pickle

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

        if "tmp_path" in locals():

            if os.path.exists(tmp_path):

                os.remove(tmp_path)
