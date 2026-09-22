# type: ignore
"""
Este archivo es para mi entrega de
'Momento de Retroalimentación: Módulo 2 Análisis y Reporte sobre el desempeño del modelo. (Portafolio Análisis)'.
Decidí hacer mi análisis con mi implementación de random forest en la que utilicé un framework.
Aquí repito mucho de mi código de la implementación con framework que hice en 'random_forest.py' y lo
modifiqué donde era necesario para esta entrega.

Pondré viñetas enumeradas para poner lo que aprendí en cada iteración, pero no dejaré el código original de cada vez que
lo corrí, solo dejaré la iteración final (que en teoría será la mejor que encuentre).
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.inspection import permutation_importance


"""
Separación y evaluación del modelo con un conjunto de prueba y un conjunto de validación (Train/Test/Validation).
"""
df = pd.read_csv("Data/Titanic-Dataset-selected-columns.csv")


def mover_target_al_final(df, posicion_target):
    columnas = df.columns.tolist()

    target = columnas.pop(posicion_target)
    columnas.append(target)

    return df[columnas]


# Posición de la columna target en el dataset original
POSICION_TARGET = 1

# Mover target a la última posición
df = mover_target_al_final(df, POSICION_TARGET)


# Separar la columna target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Es necesario indicar cuáles columnas son numéricas y cuáles categorícias para las transformaciones del pipeline
#original
"""
columnas_numericas = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]
"""
columnas_numericas = [
    "Pclass",
    "Age",
    "Fare"
]

columnas_categoricas = [
    "Sex"
]



# Separación de train, validation y test

# Primero separamos el 20% de los datos para prueba.
# Este conjunto no se utilizará durante el entrenamiento ni durante el ajuste
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Del 80% restante, separamos el 25% (equivalente al 20% del dataet original) para validación

X_entrenamiento, X_validacion, y_entrenamiento, y_validacion = train_test_split(
    X_entrenamiento,
    y_entrenamiento,
    test_size=0.25,
    random_state=42
)


# Preprocesamiento
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


# El preprocesamiento se ajusta únicamente con los datos de entrenamiento
X_entrenamiento = preprocesamiento.fit_transform(X_entrenamiento)

# Validation y Test solamente se transforman
X_validacion = preprocesamiento.transform(X_validacion)
X_prueba = preprocesamiento.transform(X_prueba)

# ==========================================
# RANDOMIZED SEARCH - RANDOM FOREST
# ==========================================

modelo_rf = RandomForestClassifier(
    random_state=42
)

# Espacio de búsqueda
parametros = {
    "n_estimators": randint(100, 501),
    "max_depth": [3, 5, 7, 10, 12, 15, 20, None],
    "min_samples_split": randint(2, 21),
    "min_samples_leaf": randint(1, 21),
    "max_features": ["sqrt", "log2", None]
}

# Búsqueda aleatoria
random_search = RandomizedSearchCV(
    estimator=modelo_rf,
    param_distributions=parametros,
    n_iter=50,
    scoring="accuracy",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_entrenamiento, y_entrenamiento)

# ==========================================
# MEJORES RESULTADOS
# ==========================================

print("==========================================")
print("MEJORES HIPERPARÁMETROS")
print("==========================================")

print(random_search.best_params_)

print("\n==========================================")
print("MEJOR ACCURACY DE VALIDACIÓN CRUZADA")
print("==========================================")

print(random_search.best_score_)

mejor_modelo = random_search.best_estimator_

# ==========================================
# IMPORTANCIA DE LAS VARIABLES
# ==========================================

# Obtener los nombres de las variables después del preprocesamiento
nombres_variables = preprocesamiento.get_feature_names_out()

# Importancia calculada por el Random Forest
importancias = pd.DataFrame({
    "variable": nombres_variables,
    "importancia": mejor_modelo.feature_importances_
})

importancias = importancias.sort_values(
    "importancia",
    ascending=False
)

print("==========================================")
print("IMPORTANCIA DE LAS VARIABLES")
print("==========================================")

print(importancias.to_string(index=False))

# Predicciones
predicciones_entrenamiento = mejor_modelo.predict(X_entrenamiento)
predicciones_validacion = mejor_modelo.predict(X_validacion)
predicciones_prueba = mejor_modelo.predict(X_prueba)

# Accuracy
accuracy_entrenamiento = accuracy_score(
    y_entrenamiento,
    predicciones_entrenamiento
)

accuracy_validacion = accuracy_score(
    y_validacion,
    predicciones_validacion
)

accuracy_prueba = accuracy_score(
    y_prueba,
    predicciones_prueba
)

print("==========================================")
print("RESULTADOS DEL MEJOR RANDOM FOREST")
print("==========================================")

print(f"Accuracy entrenamiento: {accuracy_entrenamiento:.4f}")
print(f"Accuracy validación:    {accuracy_validacion:.4f}")
print(f"Accuracy prueba:        {accuracy_prueba:.4f}")

print(f"Diferencia Train-Validation: "
      f"{accuracy_entrenamiento - accuracy_validacion:.4f}")

print(f"Diferencia Train-Test: "
      f"{accuracy_entrenamiento - accuracy_prueba:.4f}")


# Iteración 2: limitar cantidad mínima de muestras por hoja
# ==========================================
# AJUSTE DE min_samples_leaf
# ==========================================

MINIMAS_HOJA = [1, 2, 3, 4, 5, 6, 8, 10]

resultados = []


for min_hoja in MINIMAS_HOJA:

    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=min_hoja,
        max_features="sqrt",
        random_state=None
    )

    modelo.fit(X_entrenamiento, y_entrenamiento)

    # Predicciones
    predicciones_entrenamiento = modelo.predict(X_entrenamiento)
    predicciones_validacion = modelo.predict(X_validacion)
    predicciones_prueba = modelo.predict(X_prueba)

    # Accuracy
    accuracy_entrenamiento = accuracy_score(
        y_entrenamiento,
        predicciones_entrenamiento
    )

    accuracy_validacion = accuracy_score(
        y_validacion,
        predicciones_validacion
    )

    accuracy_prueba = accuracy_score(
        y_prueba,
        predicciones_prueba
    )

    # Brechas
    diferencia_train_validation = (
        accuracy_entrenamiento - accuracy_validacion
    )

    diferencia_train_test = (
        accuracy_entrenamiento - accuracy_prueba
    )

    resultados.append({
        "min_samples_leaf": min_hoja,
        "accuracy_train": accuracy_entrenamiento,
        "accuracy_validation": accuracy_validacion,
        "accuracy_test": accuracy_prueba,
        "diferencia_train_validation": diferencia_train_validation,
        "diferencia_train_test": diferencia_train_test
    })


resultados_df = pd.DataFrame(resultados)

print("==========================================")
print("RESULTADOS DEL AJUSTE DE min_samples_leaf")
print("==========================================")

print(resultados_df.to_string(index=False))


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.plot(
    resultados_df["min_samples_leaf"],
    resultados_df["accuracy_train"],
    marker="o",
    label="Train"
)

plt.plot(
    resultados_df["min_samples_leaf"],
    resultados_df["accuracy_validation"],
    marker="o",
    label="Validation"
)

plt.plot(
    resultados_df["min_samples_leaf"],
    resultados_df["accuracy_test"],
    marker="o",
    label="Test"
)

plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.title("Desempeño del Random Forest según min_samples_leaf")
plt.legend()
plt.grid(True)

plt.show()


plt.figure(figsize=(10, 6))

plt.plot(
    resultados_df["min_samples_leaf"],
    resultados_df["diferencia_train_validation"],
    marker="o",
    label="Train - Validation"
)

plt.plot(
    resultados_df["min_samples_leaf"],
    resultados_df["diferencia_train_test"],
    marker="o",
    label="Train - Test"
)

plt.axhline(0, linestyle="--")

plt.xlabel("min_samples_leaf")
plt.ylabel("Diferencia de Accuracy")
plt.title("Brecha de generalización según min_samples_leaf")
plt.legend()
plt.grid(True)

plt.show()


"""
# ==========================================
# CONFIGURACIÓN DEL RANDOM FOREST
# ==========================================

NUMERO_ARBOLES = 100
PROFUNDIDAD_MAXIMA = 10
MINIMO_MUESTRAS_DIVISION = 2
MINIMO_MUESTRAS_HOJA = 1
NUMERO_MAXIMO_CARACTERISTICAS = "sqrt"
SEMILLA = 42


modelo = RandomForestClassifier(
    n_estimators=NUMERO_ARBOLES,
    max_depth=PROFUNDIDAD_MAXIMA,
    min_samples_split=MINIMO_MUESTRAS_DIVISION,
    min_samples_leaf=MINIMO_MUESTRAS_HOJA,
    max_features=NUMERO_MAXIMO_CARACTERISTICAS,
    random_state=SEMILLA
)


# Entrenamiento con todo lo de train
modelo.fit(X_entrenamiento, y_entrenamiento)





# Evaluación
predicciones_entrenamiento = modelo.predict(X_entrenamiento)
predicciones_validacion = modelo.predict(X_validacion)
predicciones_prueba = modelo.predict(X_prueba)


accuracy_entrenamiento = accuracy_score(
    y_entrenamiento,
    predicciones_entrenamiento
)

accuracy_validacion = accuracy_score(
    y_validacion,
    predicciones_validacion
)

accuracy_prueba = accuracy_score(
    y_prueba,
    predicciones_prueba
)


print("==========================================")
print("SEPARACIÓN DE LOS DATOS")
print("==========================================")

print(f"Datos de entrenamiento: {len(X_entrenamiento)}")
print(f"Datos de validación:    {len(X_validacion)}")
print(f"Datos de prueba:        {len(X_prueba)}")


print("\n==========================================")
print("ACCURACY DEL MODELO")
print("==========================================")

print(f"Accuracy de entrenamiento: {accuracy_entrenamiento:.2%}")
print(f"Accuracy de validación:    {accuracy_validacion:.2%}")
print(f"Accuracy de prueba:        {accuracy_prueba:.2%}")


# Calcular las diferencias en accuracy
diferencia_train_validacion = (
    accuracy_entrenamiento - accuracy_validacion
)

diferencia_train_prueba = (
    accuracy_entrenamiento - accuracy_prueba
)


print("\n==========================================")
print("Diferencias de accuracy")
print("==========================================")

print(
    f"Diferencia Train-Validation: "
    f"{diferencia_train_validacion:.2%}"
)

print(
    f"Diferencia Train-Test:       "
    f"{diferencia_train_prueba:.2%}"
)



# Resultados
print("\n==========================================")
print("REPORTE DE CLASIFICACIÓN - VALIDACIÓN")
print("==========================================")

print(
    classification_report(
        y_validacion,
        predicciones_validacion
    )
)


print("\n==========================================")
print("REPORTE DE CLASIFICACIÓN - PRUEBA")
print("==========================================")

print(
    classification_report(
        y_prueba,
        predicciones_prueba
    )
)



# Matriz de confusión - Validación
clases_validacion = sorted(
    set(y_validacion) | set(predicciones_validacion)
)

matriz_validacion = confusion_matrix(
    y_validacion,
    predicciones_validacion,
    labels=clases_validacion
)

disp_validacion = ConfusionMatrixDisplay(
    confusion_matrix=matriz_validacion,
    display_labels=clases_validacion
)

disp_validacion.plot()
plt.title("Matriz de confusión - Validation")
plt.show()


# Matriz de confusión prueba

clases_prueba = sorted(
    set(y_prueba) | set(predicciones_prueba)
)

matriz_prueba = confusion_matrix(
    y_prueba,
    predicciones_prueba,
    labels=clases_prueba
)

disp_prueba = ConfusionMatrixDisplay(
    confusion_matrix=matriz_prueba,
    display_labels=clases_prueba
)

disp_prueba.plot()
plt.title("Matriz de confusión - Test")
plt.show()


# ==========================================
# GRÁFICA COMPARATIVA DE ACCURACY
# ==========================================
conjuntos = [
    "Entrenamiento",
    "Validación",
    "Prueba"
]

accuracy = [
    accuracy_entrenamiento,
    accuracy_validacion,
    accuracy_prueba
]

plt.figure(figsize=(8, 5))

plt.bar(conjuntos, accuracy)

plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Accuracy del Random Forest por conjunto de datos")

for i, valor in enumerate(accuracy):
    plt.text(
        i,
        valor + 0.02,
        f"{valor:.2%}",
        ha="center"
    )

plt.show()
"""


"""
Diagnóstico y explicación el grado de bias o sesgo: bajo medio alto
"""
# 1. Después de correr el código por primera vez
# El bias se considera bajo.

# El modelo obtiene un accuracy de 98.31% en el conjunto de entrenamiento y mantiene un desempeño de
# aproximadamente 81-82% en los conjuntos que no utilizó para entrenar. Esto indica que el modelo tiene
# una buena capacidad para aprender los patrones presentes en los datos de entrenamiento.

# Sin embargo, el accuracy de los conjuntos de validación y prueba es considerablemente menor que
# el de entrenamiento, por lo que concluyo que el problem principal del modelo es un bias alto.


"""
Diagnóstico y explicación el grado de varianza: bajo medio alto
"""
# 1. Después de correr el código por primera vez
# La varianza se considera alta.

# El modelo alcanza un accuracy de 98.31% en entrenamiento, mientras que obtiene 82.02% en validación
# y 81.01% en prueba. Esto representa una diferencia de 16.29 puntos porcentuales entre Train y Validation y de
# 17.31 puntos porcentuales entre Train y Test.

# Esta diferencia indica que el modelo tiene un desempeño considerablemente mejor sobre los
# datos con los que fue entrenado que sobre datos que no había visto, lo cual es indicativo
# de una varianza alta.


"""
Diagnóstico y explicación el nivel de ajuste del modelo: underfit fit overfit
"""
# 1. Después de correr el código por primera vez
# El modelo presenta overfitting.

# El accuracy de entrenamiento de 98.31% es muy superior al accuracy de validación de 82.02% y
# al accuracy de prueba de 81.01%. Esto indica que el Random Forest está aprendiendo los datos
# de entrenamiento con mucha precisión, pero su desempeño disminuye considerablemente al enfrentarse
# a datos nuevos.

# Los resultados de validación y prueba son relativamente cercanos wntre sí, lo que indica que la
# disminución de desempeño no se debe a una diferencia importante entre estos dos conjuntos, sino principalmente a
# la gran diferencia respecto al conjunto de entrenamiento.

# Por lo tanto, el comportamiento observado es consistente con un modelo
# que presenta overfitting y una varianza alta.



"""
Estos son códigos de prueba que usé dependiendo de la iteración
"""
"""
# Iteración 2 limitar max depth
# ==========================================
# AJUSTE DE max_depth
# ==========================================

PROFUNDIDADES = [3, 4, 5, 6, 7, 10, 12, 15, 17, 20, None]

resultados = []


for profundidad in PROFUNDIDADES:

    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=profundidad,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=None
    )

    modelo.fit(X_entrenamiento, y_entrenamiento)

    predicciones_entrenamiento = modelo.predict(X_entrenamiento)
    predicciones_validacion = modelo.predict(X_validacion)

    accuracy_entrenamiento = accuracy_score(
        y_entrenamiento,
        predicciones_entrenamiento
    )

    accuracy_validacion = accuracy_score(
        y_validacion,
        predicciones_validacion
    )

    resultados.append({
        "max_depth": profundidad,
        "accuracy_train": accuracy_entrenamiento,
        "accuracy_validation": accuracy_validacion,
        "diferencia": (
            accuracy_entrenamiento - accuracy_validacion
        )
    })


resultados_df = pd.DataFrame(resultados)

print("==========================================")
print("RESULTADOS DEL AJUSTE DE max_depth")
print("==========================================")

print(resultados_df.to_string(index=False))
"""

"""
# ==========================================
# AJUSTE DE min_samples_leaf
# ==========================================

MINIMOS_HOJA = [1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 20, 25, 30]

resultados = []


for minimo_hoja in MINIMOS_HOJA:

    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=minimo_hoja,
        max_features="sqrt",
        random_state=42
    )

    modelo.fit(X_entrenamiento, y_entrenamiento)

    predicciones_entrenamiento = modelo.predict(X_entrenamiento)
    predicciones_validacion = modelo.predict(X_validacion)

    accuracy_entrenamiento = accuracy_score(
        y_entrenamiento,
        predicciones_entrenamiento
    )

    accuracy_validacion = accuracy_score(
        y_validacion,
        predicciones_validacion
    )

    resultados.append({
        "min_samples_leaf": minimo_hoja,
        "accuracy_train": accuracy_entrenamiento,
        "accuracy_validation": accuracy_validacion,
        "diferencia": (
            accuracy_entrenamiento - accuracy_validacion
        )
    })


resultados_df = pd.DataFrame(resultados)

print("==========================================")
print("RESULTADOS DEL AJUSTE DE min_samples_leaf")
print("==========================================")

print(resultados_df.to_string(index=False))
"""