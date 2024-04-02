import typer
import pandas as pd
import logging
import os
import utils


def filter_by_frequency(df, column_name, min_frequency):
    """
    Filtra el DataFrame conservando solo las filas donde los valores de la columna especificada
    tienen una frecuencia total igual o mayor que `min_frequency`.
    """
    value_counts = df[column_name].value_counts()
    sufficient_values = value_counts[value_counts >= min_frequency].index
    return df[df[column_name].isin(sufficient_values)]

def visualize_category_frequencies(df, categories):
    """
    Visualiza las frecuencias de las categorías especificadas del DataFrame.
    """
    for category in categories:
        utils.draw_bars_categorical(df[category].value_counts(normalize=True))

def prepare_for_analysis(df, columns_to_drop, numerical_columns, categorical_df):
    """
    Prepara el DataFrame para el análisis eliminando columnas innecesarias y fusionando
    datos categóricos y numéricos.
    """
    df_filtered = df.drop(columns_to_drop, axis=1)
    df_merged = pd.merge(df[numerical_columns], df_filtered, left_index=True, right_index=True, how='inner')
    return pd.merge(df_merged, categorical_df, left_index=True, right_index=True, how='inner')

