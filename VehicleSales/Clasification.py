from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def perform_classification_with_tree(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Aunque para árboles de decisión la normalización no es necesaria, la incluimos aquí
    # para mantener la consistencia con el ejemplo anterior. Puedes optar por omitirla.
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Crear y entrenar el árbol de decisión
    tree_classifier = DecisionTreeClassifier(random_state=42)
    tree_classifier.fit(X_train, y_train)

    # Evaluar el clasificador
    y_pred = tree_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Opcional: Visualizar el árbol de decisión
    # Esto requiere la instalación de las bibliotecas graphviz y dtreeviz
    # from dtreeviz.trees import dtreeviz
    # viz = dtreeviz(tree_classifier, X_train, y_train,
    #                target_name="Target",
    #                feature_names=df.drop(target_column, axis=1).columns,
    #                class_names=list(map(str, tree_classifier.classes_)))
    # viz.view()
