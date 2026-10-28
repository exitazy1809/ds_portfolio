# preprocess.py

import logging
import unicodedata

import numpy as np
import pandas as pd


# =====================================
# Контракт признаков модели
#
# Источник истины: final_catboost_model.pkl
# (model.feature_names_, 51 признак).
# Порядок совпадает с порядком колонок X_train
# в jupyter/EDA.ipynb (ячейки 45-54).
# =====================================

TARGET_COL = "is_high_demand"

ID_COLUMNS = ["listing_id", "id"]

MODEL_FEATURES = [
    "name",
    "description",
    "host_since",
    "host_response_time",
    "host_response_rate",
    "host_acceptance_rate",
    "host_is_superhost",
    "host_listings_count",
    "host_total_listings_count",
    "host_has_profile_pic",
    "host_identity_verified",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bathrooms_text",
    "bedrooms",
    "beds",
    "minimum_nights",
    "maximum_nights",
    "minimum_minimum_nights",
    "maximum_minimum_nights",
    "minimum_maximum_nights",
    "maximum_maximum_nights",
    "minimum_nights_avg_ntm",
    "maximum_nights_avg_ntm",
    "has_availability",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "instant_bookable",
    "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms",
    "city",
    "avg_stay_nights",
    "stay_ratio",
    "name_length",
    "name_word_count",
    "desc_length",
    "bathrooms_per_guest",
    "beds_per_guest",
    "text_features",
]

# категориальные признаки CatBoost
# (model.get_cat_feature_indices(), 17 штук)
CAT_FEATURES = [
    "name",
    "description",
    "host_since",
    "host_response_time",
    "host_response_rate",
    "host_acceptance_rate",
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "neighbourhood_cleansed",
    "property_type",
    "room_type",
    "bathrooms_text",
    "has_availability",
    "instant_bookable",
    "city",
    "text_features",
]

# признаки, которые создаются в feature_engineering
ENGINEERED_FEATURES = [
    "avg_stay_nights",
    "stay_ratio",
    "name_length",
    "name_word_count",
    "desc_length",
    "bathrooms_per_guest",
    "beds_per_guest",
    "text_features",
]

# колонки, которые обязаны прийти из PostgreSQL
RAW_REQUIRED_COLUMNS = [
    col for col in MODEL_FEATURES
    if col not in ENGINEERED_FEATURES
]

RAW_NUMERIC_COLUMNS = [
    col for col in RAW_REQUIRED_COLUMNS
    if col not in CAT_FEATURES
]

# на обучении maximum_nights обрезался сверху
# (EDA.ipynb, раздел 7 Feature Engineering)
MAX_NIGHTS_CLIP = 1125

CAT_FILL_VALUE = "Unknown"


class SchemaError(ValueError):
    """Входные данные не соответствуют контракту модели."""


# =====================================
# Проверка входной схемы
# =====================================

def validate_raw_schema(df):
    """
    Проверяет, что во входном DataFrame есть
    все сырые колонки, нужные модели.
    """

    if df.empty:
        raise SchemaError("Входной датасет пустой")

    missing = [
        col for col in RAW_REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise SchemaError(
            f"Во входных данных нет колонок: {missing}"
        )

    if not any(col in df.columns for col in ID_COLUMNS):
        raise SchemaError(
            f"Во входных данных нет идентификатора: {ID_COLUMNS}"
        )


def get_listing_ids(df):
    """
    Возвращает идентификаторы объявлений:
    listing_id, если есть, иначе id.
    """

    for col in ID_COLUMNS:
        if col in df.columns:
            return df[col].copy()

    raise SchemaError(
        f"Во входных данных нет идентификатора: {ID_COLUMNS}"
    )


# =====================================
# Заполнение пропусков
# Повторяет EDA.ipynb, раздел 6.4:
# числа — медиана train, категории — "Unknown"
# =====================================

def fill_missing_values(
    df,
    fill_values,
    cat_features=CAT_FEATURES
):

    data = df.copy()

    data = data.drop(
        columns=fill_values.get("drop_columns", []),
        errors="ignore"
    )

    # числовые
    for col in RAW_NUMERIC_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            )

    numeric_fill = fill_values.get("numeric", {})

    for col, value in numeric_fill.items():
        if col in data.columns:
            data[col] = data[col].fillna(value)

    if not numeric_fill:
        nan_share = (
            data[[c for c in RAW_NUMERIC_COLUMNS if c in data.columns]]
            .isna()
            .mean()
        )
        nan_share = nan_share[nan_share > 0]

        if not nan_share.empty:
            logging.warning(
                "В fill_values нет медиан для числовых признаков. "
                "NaN передаются в CatBoost как есть "
                "(на обучении пропусков не было). Доля NaN: %s",
                nan_share.round(4).to_dict()
            )

    # категориальные
    categorical_fill = fill_values.get("categorical", {})

    for col in cat_features:
        if col in data.columns:
            data[col] = (
                data[col]
                .fillna(categorical_fill.get(col, CAT_FILL_VALUE))
                .astype(str)
            )

    return data


# =====================================
# Feature engineering
# Повторяет EDA.ipynb, разделы 7 и 8.
# Вызывается ПОСЛЕ fill_missing_values,
# как на обучении.
# =====================================

def feature_engineering(df):

    data = df.copy()

    logging.info(
        "Feature engineering start. Rows: %s",
        len(data)
    )

    # =====================================
    # 1. Условия проживания
    # =====================================

    max_nights_clean = data["maximum_nights"].clip(
        upper=MAX_NIGHTS_CLIP
    )

    data["avg_stay_nights"] = (
        data["minimum_nights"] + max_nights_clean
    ) / 2

    data["stay_ratio"] = (
        data["minimum_nights"]
        / max_nights_clean.replace(0, np.nan)
    )

    # =====================================
    # 2. Текстовые характеристики
    # =====================================

    data["name_length"] = data["name"].str.len()

    data["name_word_count"] = (
        data["name"]
        .str.split()
        .str.len()
    )

    data["desc_length"] = data["description"].str.len()

    # =====================================
    # 3. Соотношения характеристик жилья
    # =====================================

    accommodates = data["accommodates"].replace(0, np.nan)

    data["bathrooms_per_guest"] = data["bathrooms"] / accommodates

    data["beds_per_guest"] = data["beds"] / accommodates

    # =====================================
    # 4. Текстовый признак
    # На обучении передан в CatBoost
    # как категориальный, не как text feature
    # =====================================

    data["text_features"] = (
        data["name"] + " " + data["description"]
    ).apply(
        lambda x: unicodedata.normalize("NFKC", x)
    )

    logging.info(
        "FE finished. Columns: %s",
        len(data.columns)
    )

    return data


# =====================================
# Единая точка входа для инференса
# =====================================

def preprocess_for_inference(
    df,
    fill_values,
    model_features=MODEL_FEATURES,
    cat_features=CAT_FEATURES
):
    """
    Сырые данные из PostgreSQL -> матрица признаков модели.

    Порядок шагов как на обучении:
    проверка схемы -> заполнение пропусков
    -> feature engineering -> выбор признаков модели.
    """

    validate_raw_schema(df)

    data = df.copy()

    if TARGET_COL in data.columns:
        logging.info(
            "Колонка %s удалена из входных данных",
            TARGET_COL
        )
        data = data.drop(columns=[TARGET_COL])

    data = fill_missing_values(
        data,
        fill_values,
        cat_features
    )

    data = feature_engineering(data)

    missing = [
        col for col in model_features
        if col not in data.columns
    ]

    if missing:
        raise SchemaError(
            f"После FE не хватает признаков модели: {missing}"
        )

    X = data[list(model_features)].copy()

    for col in cat_features:
        X[col] = X[col].astype(str)

    non_numeric = [
        col for col in X.columns
        if col not in cat_features
        and not pd.api.types.is_numeric_dtype(X[col])
    ]

    if non_numeric:
        raise SchemaError(
            f"Числовые признаки имеют нечисловой тип: {non_numeric}"
        )

    logging.info(
        "Матрица признаков: %s, категориальных: %s",
        X.shape,
        len(cat_features)
    )

    return X
