"""
СЮДА НЕОБХОДИМО ПЕРЕНЕСТИ РЕАЛИЗОВАННЫЕ ФУНКЦИИ ИЗ ЮПИТЕР НОУТБУКА.

+ РЕАЛИЗОВАТЬ функцию preprocess_data, которая должна:
1. Обрабатывать аномальные продажи
2. Заполнять пропуски средним
3. Обогатить датасет признаками, используя функции: 
    create_temporal_features,
    create_avg_sales_feature,
    create_lag_features
    create_rolling_features
"""


import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# -----------------------------
# 1. временные признаки
# -----------------------------
def create_temporal_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year

    return df


# -----------------------------
# 2. средние продажи 
# -----------------------------
def create_avg_sales_feature(df):
    df = df.sort_values(['store', 'dept', 'date'])

    df['avg_sales_before'] = (
        df.groupby(['store', 'dept'])['weekly_sales']
        .apply(lambda x: x.expanding().mean().shift(1))
        .reset_index(level=[0,1], drop=True)
    )

    return df


# -----------------------------
# 3. лаги
# -----------------------------
def create_lag_features(df):
    df = df.sort_values(['store', 'dept', 'date'])

    df['sales_1week_ago'] = df.groupby(['store', 'dept'])['weekly_sales'].shift(1)
    df['sales_2week_ago'] = df.groupby(['store', 'dept'])['weekly_sales'].shift(2)
    df['sales_4week_ago'] = df.groupby(['store', 'dept'])['weekly_sales'].shift(4)

    return df


# -----------------------------
# 4. rolling
# -----------------------------
def create_rolling_features(df):
    df = df.sort_values(['store', 'dept', 'date'])

    grouped = df.groupby(['store', 'dept'])['weekly_sales']

    df['mean_sales_2week'] = grouped.shift(1).rolling(2).mean()
    df['mean_sales_4week'] = grouped.shift(1).rolling(4).mean()

    return df


# -----------------------------
# 5. preprocess
# -----------------------------
def preprocess_data(df):
    df = df.copy()

    # 1. аномалии 
    df['weekly_sales'] = df['weekly_sales'].clip(lower=0)

    # 2. временные признаки
    df = create_temporal_features(df)

    # 3. фичи до 
    df = create_avg_sales_feature(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)

    # 4. пропуски
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())

    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

