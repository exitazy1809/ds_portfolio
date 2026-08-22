import logging
import pandas as pd


# =====================================
# Заполнение пропусков
# Используются параметры train
# =====================================

def fill_missing_values(
    X,
    fill_values
):

    X = X.copy()

    # удаление признаков >50%

    X = X.drop(
        columns=fill_values["drop_columns"],
        errors="ignore"
    )

    # числовые

    for col, value in fill_values["numeric"].items():

        if col in X.columns:

            X[col] = X[col].fillna(
                value
            )

    # категориальные

    for col, value in fill_values["categorical"].items():

        if col in X.columns:

            X[col] = (

                X[col]
                .fillna(value)
                .astype(str)

            )

    return X


# =====================================
# Prediction
# =====================================

def run_prediction(
    model,
    X,
    ids,
    fill_values,
    cat_features
):

    logging.info(
        "Запуск инференса"
    )

    X = X.copy()

    logging.info(
        "До обработки: %s",
        X.shape
    )

    # =====================================
    # 1. Заполнение пропусков
    # =====================================

    X = fill_missing_values(
        X,
        fill_values
    )

    logging.info(
        "Пропуски обработаны"
    )

    # =====================================
    # 2. Feature Engineering
    # =====================================

    from preprocess import (
        feature_engineering
    )

    X = feature_engineering(
        X
    )

    logging.info(
        "После FE: %s",
        X.shape
    )

    # =====================================
    # 3. Категориальные признаки CatBoost
    # =====================================

    for col in cat_features:

        if col in X.columns:

            X[col] = (

                X[col]
                .fillna("Unknown")
                .astype(str)

            )

    logging.info(
        "Категории подготовлены"
    )

    # =====================================
    # 4. Сверка с моделью
    # =====================================

    model_features = (
        model.feature_names_
    )

    missing_features = (

        set(model_features)

        -

        set(X.columns)

    )

    if missing_features:

        logging.warning(
            "Добавлены отсутствующие признаки: %s",
            missing_features
        )

        for col in missing_features:

            if col in cat_features:

                X[col] = "Unknown"

            else:

                X[col] = 0

    extra_features = (

        set(X.columns)

        -

        set(model_features)

    )

    if extra_features:

        logging.warning(
            "Удалены лишние признаки: %s",
            extra_features
        )

        X = X.drop(
            columns=list(extra_features)
        )

    X = X.reindex(
        columns=model_features
    )

    logging.info(
        "Финальные признаки: %s",
        len(X.columns)
    )

    # =====================================
    # 5. Prediction
    # =====================================

    probabilities = (

        model.predict_proba(X)[:, 1]

    )

    result = pd.DataFrame(

        {

            "listing_id": ids.values,

            "score": probabilities

        }

    )

    logging.info(
        "Предсказаний: %s",
        len(result)
    )

    return result
