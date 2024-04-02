import typer
import pandas as pd
import logging
import os
import sys
sys.path.append(os.path.join(os.getcwd(),'..','VehicleSales'))

from VehicleSales.utils import (get_outliers, remove_outliers, draw_boxplot, draw_bars_categorical, draw_hist_numerical, grafico_q_q, convert_categorical_to_numeric,plot_correlation_matrix)
from VehicleSales.data_analysis import (filter_by_frequency,visualize_category_frequencies,prepare_for_analysis)
from VehicleSales.LinearRegression import perform_linear_regression
from VehicleSales.Clasification import perform_classification_with_tree

app = typer.Typer()

def setup_logging():
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    
@app.command()
def analyze_data(file_path: str = typer.Argument(..., help="Path to the car_prices.csv file")):
    setup_logging()
    logging.info("Starting analysis of vehicle sales")
    
    if not os.path.exists(file_path):
        logging.error("File path does not exist")
        raise typer.Exit(code=1)
    
    df_car_prices = pd.read_csv(file_path)
    logging.info(f"Columns in the dataset: {df_car_prices.columns}")
    
    # Describe the dataset
    logging.info(df_car_prices.describe().to_string())
    
    # Information about the dataset
    df_car_prices.info()
    
    # Null values by column
    logging.info(df_car_prices.isnull().sum().to_string())
    
    # Duplicate rows
    logging.info(f"Duplicate rows: {df_car_prices.duplicated().sum()}")
    
    # Fill missing values and drop insignificant nulls
    df_car_prices['make'] = df_car_prices['make'].fillna("other")
    df_car_prices['body'] = df_car_prices['body'].fillna("other")
    df_car_prices['transmission'] = df_car_prices['transmission'].fillna("other")
    df_cleaned = df_car_prices.dropna(subset=['year', 'make', 'color', 'transmission', 'odometer', 'mmr', 'sellingprice', 'saledate'])
    
    # Convert 'saledate' to datetime
    df_cleaned['sale_date'] = pd.to_datetime(df_cleaned['saledate'])
    
    # Filtrar por frecuencia mínima
    df_cleaned['make'] = df_cleaned['make'].str.lower()
    df_cleaned['color'] = df_cleaned['color'].str.lower()

    df_filtered = filter_by_frequency(df_cleaned, 'make', 15000)
    df_filtered = filter_by_frequency(df_filtered, 'color', 5000)

    # Visualizar frecuencias de categorías filtradas
    visualize_category_frequencies(df_filtered, ['make', 'color', 'transmission'])

    # Preparar DataFrame para análisis posteriores TODO! ESTO ESTA MAL PORQUE 
    # NO SE USA CORRECTAMENTE EL MERGE, LE PASAMOS DOS VECES EL df_Filtered
    columns_to_drop = ['model','trim','vin','state','body', 'interior', 'saledate', 'seller','condition','sale_date']
    numerical_cols = df_filtered.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df_filtered.select_dtypes(include=[object]).columns.tolist()
    #df_for_analysis = prepare_for_analysis(df_filtered, columns_to_drop, numerical_cols, df_filtered)
    
    #=========================================================
    #               Distribucion de las variables
    #=========================================================
    # Handle categorical and numerical columns
    df_filtered = df_filtered.drop(columns_to_drop,axis=1) 
    df_columnas_numericas = df_filtered.select_dtypes(include=['float', 'int'])
    df_columns_categorical = df_filtered.select_dtypes(include=[object])
    df_for_analysis = pd.merge(df_columnas_numericas, df_columns_categorical, left_index=True, right_index=True)
    numerical_cols = df_for_analysis.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df_for_analysis.select_dtypes(include=[object]).columns.tolist()
    
    # Visualizations
    #AQUI QUITAR LAS QUE SON MUY GRANDES BODY SELLINGPRICE ETC
    draw_boxplot(df_for_analysis[numerical_cols])
    for col in categorical_cols:
        count_normalized = df_for_analysis[col].str.lower().value_counts(normalize=True)
        draw_bars_categorical(count_normalized)
    
    #=========================================================
    #               OUTLIERS
    #=========================================================
    # Outlier removal and further analysis
    # Example for 'year' column
    df_without_outliers = get_outliers(df_for_analysis, 'year', 3)
    
    #=========================================================
    #               MATRIZ DE CORRELACION
    #=========================================================
    
    # Conversión de variables categóricas a numéricas
    categorical_columns = ['make', 'transmission', 'color']  # Ajustar según sea necesario
    df_final = convert_categorical_to_numeric(df_without_outliers, categorical_columns)

    # Visualizar la matriz de correlación para todas las variables
    plot_correlation_matrix(df_final, "Matriz de Correlación con Variables Categóricas")

    # Visualizar la matriz de correlación solo para variables numéricas originales
    numerical_columns = [col for col in df_without_outliers.columns if col not in categorical_columns]  # Ajusta según tus columnas numéricas originales
    plot_correlation_matrix(df_without_outliers[numerical_columns], "Matriz de Correlación para Variables Numéricas")

    logging.info("Analysis completed successfully.")
    
    #=========================================================
    #               Linear Regression
    #=========================================================
    perform_linear_regression(df_without_outliers[numerical_columns], target_column='sellingprice')
    
if __name__ == "__main__":
    app()
