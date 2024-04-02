import pandas as pd
import numpy as np
import logging
import os
import scipy.stats as stats
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from VehicleSales.utils import *
import typer
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

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
    
    #=========================================================
    #               Distribucion de las variables
    #=========================================================
    # >>>>>>>>>>>>>>>    Columnas numericas
    df_columnas_numericas = df_sin_nulos.select_dtypes(include=['float', 'int'])
    df_columnas_numericas['year'][get_outliers(df_columnas_numericas,name_column='year',umbral=3)]
    
    #Boxplot columnas numericas
    draw_boxplot(df_columnas_numericas)
    #Despues de visualizar los boxplot, observo que en las variables: odometer, sellingprice pueden existir outliers
    # >>>>>>>>>>>>>>>    Columnas categoricas
    df_columns_categorical = df_sin_nulos.select_dtypes(include=[object])
    # Usamos el value_counts(): para contar todos los valores unicos de cada variable en una columna.
    # Para ignorar mayúsculas y minúsculas para contar todas 
    # las variantes de una marca como iguales,
    # primero convertir toda la columna a minúsculas usando str.lower() 
    # y luego aplicar value_counts()
    conteo_marcas_normalizado = df_columns_categorical['make'].str.lower().value_counts()
    conteo_body_normalizado = df_columns_categorical['body'].str.lower().value_counts()
    conteo_transmission_normalizado = df_columns_categorical['transmission'].str.lower().value_counts()
    conteo_color_normalizado = df_columns_categorical['color'].str.lower().value_counts()
    conteo_interior_normalizado = df_columns_categorical['interior'].str.lower().value_counts()
    conteo_seller_normalizado = df_columns_categorical['seller'].str.lower().value_counts()
    
    draw_bars_categorical(conteo_marcas_normalizado)
    draw_bars_categorical(conteo_body_normalizado)
    draw_bars_categorical(conteo_transmission_normalizado)
    draw_bars_categorical(conteo_color_normalizado)
    draw_bars_categorical(conteo_interior_normalizado)
    #draw_bars_categorical(conteo_seller_normalizado)
    

    #=========================================================
    #               Variables con pocos valores
    #=========================================================
    # Decido filtrar por la variable make, color, porque desde mi punto de vista es la mas
    # relevante.
    # Filtrar 'conteo_marcas_normalizado' para obtener solo las marcas que cumplen con el umbral
    marcas_suficientes = conteo_marcas_normalizado[conteo_marcas_normalizado >= 15000].index
    
    #  Ahora, queremos conservar solo las filas de 'df_columns_categorical' cuyas marcas están en 'marcas_suficientes'
    df_filtrado = df_columns_categorical[df_columns_categorical['make'].str.lower().isin(marcas_suficientes)]
    
    color_suficientes = conteo_color_normalizado[conteo_color_normalizado >= 5000].index
    df_filtrado = df_filtrado[df_filtrado['color'].str.lower().isin(color_suficientes)]

    #Repito el conteo anterior y los graficos de barras
    conteo_marcas_normalizado = df_filtrado['make'].str.lower().value_counts()
    conteo_body_normalizado = df_filtrado['body'].str.lower().value_counts()
    conteo_transmission_normalizado = df_filtrado['transmission'].str.lower().value_counts()
    conteo_color_normalizado = df_filtrado['color'].str.lower().value_counts()
    conteo_interior_normalizado = df_filtrado['interior'].str.lower().value_counts()
    conteo_seller_normalizado = df_filtrado['seller'].str.lower().value_counts()
    
    draw_bars_categorical(conteo_marcas_normalizado)
    draw_bars_categorical(conteo_body_normalizado)
    draw_bars_categorical(conteo_transmission_normalizado)
    draw_bars_categorical(conteo_color_normalizado)
    draw_bars_categorical(conteo_interior_normalizado)
    
    #Despues de analizar los graficos elimino las variables interior y body
    df_filtrado = df_filtrado.drop(['body','interior','saledate','seller'],axis=1)
    
    #Unimos las variables categoricas y numericas
    df_merged = pd.merge(df_columnas_numericas, df_filtrado, left_index=True, right_index=True)
    
    #=========================================================
    #               OUTLIERS
    #=========================================================
    # dibujamos los boxplot del dataframe filtrado
    draw_boxplot(df_columnas_numericas)
    
    #Eliminar outliers
    df_sin_outliers = df_columnas_numericas.copy()
    df_sin_outliers = get_outliers(df_sin_outliers,name_column='year',umbral=3)
    df_sin_outliers = get_outliers(df_sin_outliers,name_column='odometer',umbral=3)
    df_sin_outliers = get_outliers(df_sin_outliers,name_column='mmr',umbral=3)
    df_sin_outliers = get_outliers(df_sin_outliers,name_column='sellingprice',umbral=3)
    #=========================================================
    #               MATRIZ DE CORRELACION
    #=========================================================
    df_merged = pd.merge(df_sin_outliers, df_filtrado, left_index=True, right_index=True)

    # Convertir los valores categoricos en numericos
    # para añadir las categoricas es necesario convertirlas con one-hot encoding
    one_hot_df_make = pd.get_dummies(df_merged['make'], prefix='make_onehot')
    one_hot_df_transmission = pd.get_dummies(df_merged['transmission'], prefix='transmi_onehot')
    one_hot_df_color = pd.get_dummies(df_merged['color'], prefix='color_onehot')
    df_cate_num = pd.concat([df_merged, one_hot_df_make,one_hot_df_transmission,one_hot_df_color], axis=1)
    df_cate_num = df_cate_num.drop(['make','transmission','color','fecha venta'],axis=1)
    # Calcular la matriz de correlación
    # Asumiendo que df_columnas_numericas es tu DataFrame y ya contiene solo datos numéricos
    matriz_correlacion_cate_nume = df_cate_num.corr()

    # Usar seaborn para crear el mapa de calor
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_correlacion_cate_nume, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title('Mapa de Calor de la Matriz de Correlación con Valores')
    plt.show()
    #Segun la matriz de todas las variables, queda muy claro que las variables que mas correlacion
    # tienen son las numericas, por tanto hacemos una matriz solo de las numericas.
    # Asumiendo que df_columnas_numericas es tu DataFrame y ya contiene solo datos numéricos
    matriz_correlacion_nume = df_sin_outliers.corr()

    # Usar seaborn para crear el mapa de calor
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_correlacion_nume, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title('Mapa de Calor de la Matriz de Correlación con Valores')
    plt.show()
    #Despues de visulizar la matriz de valores numericos, 
    # elimino la variable mmr porque esta muy correlacionada con el precio
    df_sin_outliers = df_sin_outliers.drop(['mmr'],axis=1)
    

    #=========================================================
    #               Regresion lineal simple
    #=========================================================
    columns = ['year', 'odometer',  'make_onehot_BMW',
    'make_onehot_Chevrolet', 'make_onehot_Chrysler', 'make_onehot_Dodge',
    'make_onehot_Ford', 'make_onehot_Honda', 'make_onehot_Hyundai',
    'make_onehot_Infiniti', 'make_onehot_Jeep', 'make_onehot_Kia',
    'make_onehot_Mercedes-Benz', 'make_onehot_Nissan', 'make_onehot_Toyota',
    'make_onehot_bmw', 'make_onehot_chevrolet', 'make_onehot_chrysler',
    'make_onehot_dodge', 'make_onehot_ford', 'make_onehot_honda',
    'make_onehot_hyundai', 'make_onehot_jeep', 'make_onehot_kia',
    'make_onehot_nissan', 'make_onehot_toyota', 'transmi_onehot_automatic',
    'transmi_onehot_manual', 'transmi_onehot_otro', 'color_onehot_beige',
    'color_onehot_black', 'color_onehot_blue', 'color_onehot_brown',
    'color_onehot_burgundy', 'color_onehot_gold', 'color_onehot_gray',
    'color_onehot_green', 'color_onehot_red', 'color_onehot_silver',
    'color_onehot_white', 'color_onehot_—']
    X = df_cate_num[['odometer']]
    y = df_cate_num['sellingprice']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizar las variables
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Crear un modelo de regresión lineal
    model = LinearRegression()

    # Entrenar el modelo
    model.fit(X_train_scaled, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = model.predict(X_test_scaled)

    # Imprimir los coeficientes y el intercepto
    print('Coeficientes:', model.coef_)
    print('Intercepto:', model.intercept_)

    # Calcular la precisión del modelo
    print('Precisión del modelo:', metrics.r2_score(y_test, y_pred))
    #Si existe sesgo, equilibrar los conjuntos sobre el mas pequeño
    
    # Crear una regresion linear simple, solo con la marca y el precio
    
    # Generar una regresion linear multiple, con varios valores
if __name__ == "__main__":
    main()