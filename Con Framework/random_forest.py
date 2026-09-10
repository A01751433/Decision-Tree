# type:ignore
"""
Este es mi archivo para la entrega:
'Momento de Retroalimentación: Módulo 2 Uso de framework o biblioteca de aprendizaje máquina
para la implementación de una solución. (Portafolio Implementación)'

Decidí implementar Random Forest
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report
)
import matplotlib.pyplot as plt


def mover_target_al_final(df, posicion_target):
    columnas = df.columns.tolist()

    target = columnas.pop(posicion_target)
    columnas.append(target)

    return df[columnas]


df = pd.read_csv("Data/Titanic-Dataset-selected-columns.csv")

# Posición de la columna target en el dataset original
POSICION_TARGET = 1

# Mover target a la última posición
df = mover_target_al_final(df, POSICION_TARGET)


# Separar la columna target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

"""
Es necesario poner aquí a mano el nombre de las columnas para que funcione bien
"""
# Decidí no incluir PassengerId, Name y Ticket porque me parece que no son predictores de ninguna manera
# relevante de la columna target
columnas_numericas = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

columnas_categoricas = [
    "Sex"
]

# Dividir los datos en entrenamiento y prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# One Hot Encoding para las columnas categóricas y manejo de valores faltantes
preprocesamiento = ColumnTransformer(
    transformers=[
        (
            "numericas",
            SimpleImputer(strategy="median"),
            columnas_numericas
        ),
        (
            "categoricas",
            Pipeline([
                ("imputador", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            columnas_categoricas
        )
    ]
)

"""
==========================================
CONFIGURACIÓN DEL RANDOM FOREST
==========================================
"""
NUMERO_ARBOLES = 100
PROFUNDIDAD_MAXIMA = None
MINIMO_MUESTRAS_DIVISION = 2
MINIMO_MUESTRAS_HOJA = 1
NUMERO_MAXIMO_CARACTERISTICAS = "sqrt"
SEMILLA = 42

# Crear el Random Forest
modelo = RandomForestClassifier(
    n_estimators=NUMERO_ARBOLES,
    max_depth=PROFUNDIDAD_MAXIMA,
    min_samples_split=MINIMO_MUESTRAS_DIVISION,
    min_samples_leaf=MINIMO_MUESTRAS_HOJA,
    max_features=NUMERO_MAXIMO_CARACTERISTICAS,
    random_state=SEMILLA
)


# Preprocesar los datos
X_entrenamiento = preprocesamiento.fit_transform(X_entrenamiento)
X_prueba = preprocesamiento.transform(X_prueba)


# Entrenar el modelo
modelo.fit(X_entrenamiento, y_entrenamiento)


# Realizar predicciones
predicciones = modelo.predict(X_prueba)


# ==========================================
# ESTADÍSTICAS DEL MODELO
# ==========================================

# Accuracy de entrenamiento
predicciones_entrenamiento = modelo.predict(X_entrenamiento)

accuracy_entrenamiento = accuracy_score(
    y_entrenamiento,
    predicciones_entrenamiento
)


# Accuracy de prueba
accuracy_prueba = accuracy_score(
    y_prueba,
    predicciones
)


# Diferencia entre entrenamiento y prueba
diferencia_accuracy = accuracy_entrenamiento - accuracy_prueba


print("Datos de entrenamiento:", len(X_entrenamiento))
print("Datos de prueba:", len(X_prueba))
print("\n==========================================")
print("ESTADÍSTICAS DEL MODELO")
print("==========================================")

print(f"Accuracy de entrenamiento: {accuracy_entrenamiento:.2%}")
print(f"Accuracy de prueba:        {accuracy_prueba:.2%}")
print(f"Diferencia:                {diferencia_accuracy:.2%}")


# ==========================================
# REPORTE DE CLASIFICACIÓN
# ==========================================

print("\n==========================================")
print("REPORTE DE CLASIFICACIÓN")
print("==========================================")

print(classification_report(y_prueba, predicciones))


# ==========================================
# MATRIZ DE CONFUSIÓN
# ==========================================

# Obtener las clases presentes
clases = sorted(set(y_prueba) | set(predicciones))


# Matriz de confusión
matriz = confusion_matrix(
    y_prueba,
    predicciones,
    labels=clases
)


# Mostrar matriz de confusión
disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=clases
)

disp.plot()
plt.title("Matriz de confusión - Random Forest")
plt.show()