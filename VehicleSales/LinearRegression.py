from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

import pandas as pd

def perform_linear_regression(df, target_column):
    # Separar las variables independientes (X) de la dependiente (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Normalizar los datos
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Realizar predicciones y evaluar el modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    
# Imprimir los coeficientes y el intercepto
    print('Coeficientes:', model.coef_)
    print('Intercepto:', model.intercept_)

    # Calcular la precisión del modelo
    print('Precisión del modelo:', metrics.r2_score(y_test, y_pred))


# Suponiendo que 'df_final' es tu DataFrame final preparado para análisis
# perform_linear_regression(df_final, 'tu_columna_objetivo')
