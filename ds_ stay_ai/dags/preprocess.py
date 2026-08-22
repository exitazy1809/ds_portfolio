# preprocess.py

import logging

import numpy as np
import pandas as pd


# =====================================
# Признаки, которые удаляются
# одинаково train + inference
# =====================================

DROP_COLS = [

    # идентификаторы
    "id",
    "host_id",
    "source",

    # текстовые поля
    "name",
    "description",
    "review_text",

    # высокая кардинальность
    "amenities",
    "host_verifications",

    # target leakage
    "median_reviews_ltm",
    "median_price",
    "median_rating",
    "price_ratio",
    "rating_ratio",

    # заменяется на host_age_days
    "host_since"

]


def feature_engineering(
    df,
    inference_date=None
):
    """
    Feature engineering.

    ВАЖНО:
    - пропуски НЕ заполняются
    - медианы НЕ считаются
    - правила заполнения НЕ создаются

    Заполнение выполняется через:
    fill_values.pkl
    """

    data = df.copy()

    logging.info(
        "Feature engineering start. Rows: %s",
        len(data)
    )

    # =====================================
    # 1. Приведение типов
    # =====================================

    numeric_cols = [

        "price",
        "accommodates",
        "minimum_nights",
        "maximum_nights",
        "bathrooms",
        "beds"

    ]

    for col in numeric_cols:

        if col in data.columns:

            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            )

    # =====================================
    # 2. Цена
    # =====================================

    if {
        "price",
        "accommodates"

    }.issubset(data.columns):

        data["price_per_person"] = (

            data["price"]

            /

            data["accommodates"]
            .replace(
                0,
                np.nan
            )

        )

        data["log_price"] = np.log1p(
            data["price"]
        )

    # =====================================
    # 3. Длительность проживания
    # =====================================

    if {
        "minimum_nights",
        "maximum_nights"

    }.issubset(data.columns):

        data["avg_stay_nights"] = (

            data["minimum_nights"]

            +

            data["maximum_nights"]

        ) / 2

        data["stay_ratio"] = (

            data["minimum_nights"]

            /

            data["maximum_nights"]
            .replace(
                0,
                np.nan
            )

        )

    # =====================================
    # 4. Возраст хоста
    # =====================================

    if "host_since" in data.columns:

        data["host_since"] = pd.to_datetime(

            data["host_since"],

            errors="coerce"

        )

        if inference_date is not None:

            inference_date = pd.to_datetime(
                inference_date
            )

            data["host_age_days"] = (

                inference_date

                -

                data["host_since"]

            ).dt.days

        else:

            data["host_age_days"] = (

                pd.Timestamp.today()

                -

                data["host_since"]

            ).dt.days

    # =====================================
    # 5. Качество хоста
    # =====================================

    if "host_is_superhost" in data.columns:

        superhost_flag = (

            data["host_is_superhost"]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": 1,
                    "false": 0
                }
            )
            .fillna(0)

        )

    else:

        superhost_flag = 0

    if "host_identity_verified" in data.columns:

        verified_flag = (

            data["host_identity_verified"]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": 1,
                    "false": 0
                }
            )
            .fillna(0)

        )

    else:

        verified_flag = 0

    if "host_response_rate" in data.columns:

        response_rate = (

            data["host_response_rate"]
            .astype(str)
            .str.replace(
                "%",
                "",
                regex=False
            )

        )

        response_rate = (

            pd.to_numeric(
                response_rate,
                errors="coerce"
            )
            .fillna(0)
            /
            100

        )

    else:

        response_rate = 0

    data["host_quality_score"] = (

        superhost_flag * 0.4

        +

        verified_flag * 0.3

        +

        response_rate * 0.3

    )

    # =====================================
    # 6. Текстовые признаки
    # =====================================

    if "name" in data.columns:

        data["name_length"] = (

            data["name"]
            .fillna("")
            .astype(str)
            .str.len()

        )

        data["name_word_count"] = (

            data["name"]
            .fillna("")
            .astype(str)
            .str.split()
            .str.len()

        )

    if "description" in data.columns:

        data["desc_length"] = (

            data["description"]
            .fillna("")
            .astype(str)
            .str.len()

        )

    if "amenities" in data.columns:

        data["amenities_count"] = (

            data["amenities"]
            .fillna("")
            .astype(str)
            .apply(
                lambda x:
                len(x.split(","))
            )

        )

    # =====================================
    # 7. Ratio признаки
    # =====================================

    if {
        "bathrooms",
        "accommodates"

    }.issubset(data.columns):

        data["bathrooms_per_guest"] = (

            data["bathrooms"]

            /

            data["accommodates"]
            .replace(
                0,
                np.nan
            )

        )

    if {
        "beds",
        "accommodates"

    }.issubset(data.columns):

        data["bedrooms_per_guest"] = (

            data["beds"]

            /

            data["accommodates"]
            .replace(
                0,
                np.nan
            )

        )

    # =====================================
    # 8. Удаление лишних колонок
    # =====================================

    data = data.drop(

        columns=DROP_COLS,

        errors="ignore"

    )

    logging.info(
        "FE finished. Columns: %s",
        len(data.columns)
    )

    logging.info(
        "Final features: %s",
        list(data.columns)
    )

    return data


def preprocess_for_inference(
    df,
    inference_date=None
):

    return feature_engineering(
        df,
        inference_date
    )
