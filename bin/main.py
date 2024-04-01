import pandas as pd
import numpy as np
import logging
import os
import scipy.stats as stats
from matplotlib import pyplot as plt
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

def get_outliers(df,name_column:str,umbral:int):
    media = df[name_column].mean()
    std_dev = df[name_column].std()

    outliers = ((df[name_column]- media).abs() > umbral * std_dev)
    print(df[name_column][outliers])
    return outliers

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

def main():
    logging.info(f"Main of vehicle sales")
    df_car_prices = pd.read_csv(os.path.join(os.getcwd(),'..','data','car_prices.csv'))
    print(f"El fichero es:{df_car_prices.columns}")
    
    #0.Describe
    print(f"{df_car_prices.describe()}")
    
    #1.Recoger la informacion de todo el fichero
    #para analizar la cantidad de datos no nulos.
    print(f"{df_car_prices.info()}")
    
    #2. Datos nulos por columnas
    print(f"{df_car_prices.isnull().sum()}")
    
    #3. Datos duplicados
    print(f"{df_car_prices.duplicated().sum()}")
    
    #4. Que hacer con los datos que faltan
    # En este caso la columna transmission es la que mas valores
    # faltantes tiene y parece que puede ser automatica o manual
    # faltan: 65352 total:558837 entonces faltan: 11%
    
    # Para determinar el precio, voy a considerar relevante
    # year, make (marca), model, body, transmision, odometria(distancia recorrida),
    # color, interior, seller, mmr (maxima carga que puede remolcar),
    # precio, fecha de venta
    columnas = ['year', 'make','body','transmission','odometer',
                'color', 'interior', 'seller','mmr', 'sellingprice', 'saledate']
    df = df_car_prices[columnas]

    # Vamos a rellenar los valores nulos
    # make: otro 
    # body: otro
    # transmission: otro
    # el resto de valores nulos son insignifiantes por tanto los voy a eliminar
    df['make'] = df['make'].fillna("otro")
    df['body'] = df['body'].fillna("otro")
    df['transmission'] = df['transmission'].fillna("otro")
    df_sin_nulos = df.dropna()
    
    #La columna de sale date la convertimos en fecha 
    df_sin_nulos['fecha venta'] = pd.to_datetime(df_sin_nulos['saledate'])
    
    #DETECTAR VALORES ANOMALOS
    # - Columnas numericas
    df_columnas_numericas = df_sin_nulos.select_dtypes(include=['float', 'int'])
    df_columnas_numericas['year'][get_outliers(df_columnas_numericas,name_column='year',umbral=3)]
    
    #Boxplot columnas numericas
    draw_boxplot(df_columnas_numericas)
    #Despues de visualizar los boxplot, observo que en las variables: odometer, sellingprice pueden existir outliers
    # - Columnas categoricas
    df_columns_categorical = df_sin_nulos.select_dtypes(include=[object])
    # 1. Elegir una columna y sacar todos los valores unicos
    #marcas = np.unique(df_sin_nulos['make'])
    conteo_marcas_normalizado = df_columns_categorical['make'].str.lower().value_counts()
    conteo_body_normalizado = df_columns_categorical['body'].str.lower().value_counts()
    conteo_transmission_normalizado = df_columns_categorical['transmission'].str.lower().value_counts()
    conteo_color_normalizado = df_columns_categorical['color'].str.lower().value_counts()
    conteo_interior_normalizado = df_columns_categorical['interior'].str.lower().value_counts()
    conteo_seller_normalizado = df_columns_categorical['seller'].str.lower().value_counts()
    
    
    
    # antes de eliminarlos visualizo la distribucion de estos parametros
    for col in df_columnas_numericas.columns:
        fig, ax = plt.subplots(figsize=(8,6))
        ax.hist(df_columnas_numericas[col], edgecolor='black')
        ax.set_xlabel('Valores')
        ax.set_ylabel('Frecuencia')
        ax.set_title(f'Histograma de la columna {col}')
        plt.show()
        
    # outliers eliminar los outliers 
    #Si existe sesgo, equilibrar los conjuntos sobre el mas pequeño
    
    # Crear una regresion linear simple, solo con la marca y el precio
    
    # Generar una regresion linear multiple, con varios valores
if __name__ == "__main__":
    main()