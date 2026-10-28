import logging

import pandas as pd

from preprocess import SchemaError


# =====================================
# Сверка артефактов с моделью
# =====================================

def get_model_cat_features(model):
    """
    Категориальные признаки, с которыми обучена модель.
    Берутся из самой модели, а не из cat_features.pkl.
    """

    names = list(model.feature_names_)

    return [
        names[i]
        for i in model.get_cat_feature_indices()
    ]


def check_artifacts_consistency(
    model,
    expected_features,
    cat_features_artifact=None
):
    """
    Проверяет, что модель из S3 соответствует
    контракту признаков в коде (preprocess.MODEL_FEATURES).
    """

    model_features = list(model.feature_names_)

    if model_features != list(expected_features):

        missing = sorted(set(model_features) - set(expected_features))
        extra = sorted(set(expected_features) - set(model_features))

        raise SchemaError(
            "Признаки модели не совпадают с контрактом в коде. "
            f"Нет в коде: {missing}; нет в модели: {extra}; "
            "или отличается порядок."
        )

    model_cat_features = get_model_cat_features(model)

    if (
        cat_features_artifact is not None
        and list(cat_features_artifact) != model_cat_features
    ):
        logging.warning(
            "cat_features.pkl не совпадает с категориальными "
            "признаками модели. Используются признаки модели: %s",
            model_cat_features
        )

    return model_cat_features


# =====================================
# Prediction
# =====================================

def run_prediction(
    model,
    X,
    ids
):
    """
    Batch-инференс на уже подготовленной матрице признаков.
    Предобработка выполняется в preprocess.preprocess_for_inference.
    """

    logging.info(
        "Запуск инференса: %s",
        X.shape
    )

    model_features = list(model.feature_names_)

    if list(X.columns) != model_features:
        raise SchemaError(
            "Колонки X не совпадают с model.feature_names_ "
            "по составу или порядку"
        )

    if len(X) != len(ids):
        raise ValueError(
            f"Число строк X ({len(X)}) "
            f"не совпадает с числом id ({len(ids)})"
        )

    probabilities = model.predict_proba(X)[:, 1]

    result = pd.DataFrame(
        {
            "listing_id": ids.values,
            "score": probabilities
        }
    )

    logging.info(
        "Предсказаний: %s, средний score: %.4f",
        len(result),
        result["score"].mean()
    )

    return result
