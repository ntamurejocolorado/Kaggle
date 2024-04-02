import pandas as pd
import scipy.stats as stats
from matplotlib import pyplot as plt
import seaborn as sns

def get_outliers(df,name_column:str,umbral:int):
    media = df[name_column].mean()
    std_dev = df[name_column].std()

    outliers = ((df[name_column]- media).abs() > umbral * std_dev)
    #print(df[name_column][outliers])
    df_no_outliers = df[~outliers]
    return df_no_outliers

def remove_outliers(df,name_column:str):
    zscores = stats.zscore(df[name_column])
    # Crear una máscara para seleccionar los outliers (más allá de 2 o -2)
    outlier_mask = (zscores > 2) | (zscores < -2)

    # Filtrar el DataFrame original para obtener solo los valores sin outliers
    df_no_outliers = df[~outlier_mask]
    return df_no_outliers

def draw_boxplot(df_columnas_numericas):
    #Obtener graficos con las distribuciones por year,marca,odometer,color, transmision
    # Obtén los nombres de las columnas
    # Crea los boxplots utilizando las columnas del dataframe
    for col in df_columnas_numericas.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(df_columnas_numericas[col], patch_artist=True)
        ax.set_title(f"Boxplot para {col}")
        ax.set_ylabel("Valores")
        plt.show()
        
def draw_bars_categorical(conteo_normalizado):
    fig = plt.figure()
    plt.bar(conteo_normalizado.index, conteo_normalizado)
    plt.xticks(rotation=45, ha="right")
    plt.show()
    
def draw_hist_numerical(df_columnas_numericas):
    # antes de eliminarlos visualizo la distribucion de estos parametros
    for col in df_columnas_numericas.columns:
        fig, ax = plt.subplots(figsize=(8,6))
        ax.hist(df_columnas_numericas[col], edgecolor='black')
        ax.set_xlabel('Valores')
        ax.set_ylabel('Frecuencia')
        ax.set_title(f'Histograma de la columna {col}')
        plt.show()
        
def grafico_q_q(df, colum:str):
    stats.probplot(df[colum], dist="norm", plot=plt)
    plt.title('Gráfico Q-Q')
    plt.show()
    
def convert_categorical_to_numeric(df, categorical_columns):
    """
    Convierte las columnas categóricas en columnas numéricas mediante one-hot encoding.
    Devuelve un nuevo DataFrame con las columnas originales y las nuevas columnas numéricas.
    """
    for column in categorical_columns:
        dummies = pd.get_dummies(df[column], prefix=f"{column}_onehot")
        df = pd.concat([df, dummies], axis=1)
    df = df.drop(categorical_columns, axis=1)
    return df

def plot_correlation_matrix(df, title="Matriz de Correlación"):
    """
    Calcula y visualiza la matriz de correlación para el DataFrame proporcionado.
    """
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title(title)
    plt.show()