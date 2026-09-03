# type:ignore

from collections import Counter
import math
import pandas as pd
import pprint
import random

"""
- Para fines de que yo lo probara y puediera exportar esto sin necesidad de compartir un .csv
  decidí incluir los datos de entrada directamente en el código.
- Estos son los mismos datos que los de Tema 7 - Tree Exercise Empty
- Mientras estaba escribiendo esto me di cuenta de que el nombre de un par de variables y la implementación de
  algunas funciones dependen de mi dataset específico, en caso de ser necesario las puedo cambiar luego.
"""
df = pd.read_csv("Data/car_data.csv")
#df = pd.read_csv("Data/dataset_compra_coche.csv")
titulo_columnas = df.columns.tolist()
datos = df.values.tolist()
#print(dataset)
# Las columnas son Outlook, Temperature, Humidity, Wind, Play

"""
Función que divide el dataset en dos partes, por defecto divide 70/30 pero se puede cambiar le porcentaje de datos
de entrenamiento con el segundo parámetro.
La función regresa dos listas, la primera es la de entrenamiento y la segunda la de prueba.
"""
def dividir_datos(data, porcentaje_entrenamiento=0.7):
    datos = data.copy()
    random.shuffle(datos)

    cantidad_entrenamiento = int(len(datos) * porcentaje_entrenamiento)

    datos_entrenamiento = datos[:cantidad_entrenamiento]
    datos_prueba = datos[cantidad_entrenamiento:]

    return datos_entrenamiento, datos_prueba

dataset, datos_prueba = dividir_datos(datos)

# Primero voy a hacer una función que cuente la cantidad de valores diferentes en cada columna
def valores(data):
    # No sé qué tan inneficiente sea esto, pero la forma más sencilla es hacer comprensión de listas y contar los resultados ahí
    i = 0
    contadores = []
    while i < len(data[0]):
        columna = [fila[i] for fila in data]
        conteo = Counter(columna)
        contadores.append(conteo)
        i += 1
    return contadores


#prueba = valores(dataset)
#print(prueba)
#print(prueba[0]["Sunny"])
#print(prueba[len(dataset[0]) - 1].keys())


"""
Esta función es para contar cuántas veces ocurre cada valor de una columna en contraste con los valores de la columna "Y"
Esta función requiere de que la columna de los valores "Y" sea la última columna del dataset
Hace su operación solo para una columna por llamada, hice esto porque no sabía cómo estructurar los datos, pensando
que cada columna puede tener una cantidad diferente de valores únicos y que creo que todas las operaciones que siguen
se realizan por columna y no con múltiples columnas.
"""
# Hasta donde puedo discernir, el órden de los valores (si va primero el Yes o el No) no afecta ninguna operación
# creo que todas las operaciones se hacen igual con ambos valores (o todos los valores si es que hay más de 2)
# Esto de y_keys se usa para poder tener valores default de 0 para poder hacer las operaciones luego
y_keys = {}
values = valores(dataset)
for i in values[len(dataset[0]) - 1].keys():
    y_keys.setdefault(i, 0)

"""def contar_play(dataset, columna):
    conteo = {}

    for fila in dataset:
        valor = fila[columna]
        # Esta líneasignifica que la función no funciona si una columna no tiene nombre
        play = fila[len(titulo_columnas) - 1]

        if valor not in conteo:
            conteo[valor] = Counter(y_keys)

        conteo[valor][play] += 1

    return conteo
    """

def contar_play(data, columna):

    conteo = {}

    for fila in data:
        valor = fila[columna]
        clase = fila[-1]

        if valor not in conteo:
            conteo[valor] = Counter()

        conteo[valor][clase] += 1

    return conteo

#prueba = contar_play(dataset, 0)
#print(prueba)
#print(prueba["Sunny"]["Yes"])  # 2

"""
Es una función que calcula la entropía para una sola categoría. La función de contar_play regresa un counter, esta función calcula la entropía
para una de las categoría de ese objeto, cuál se especifica indicando el índice de la categoría deseada.
Nota: no sé muy bien porqué hay una entropía que tiene que ser positiva, decidí que esta función se queda así (creo que siempre da valores negativos)
y en caso de ser necesario haré el valor en un positivo donde se ocupe.
"""
"""
def entropia(conteo, categoria):

    llave = list(conteo.keys())[categoria]
    no = conteo[llave]["No"]
    yes = conteo[llave]["Yes"]
    # Si yes o no son iguales a 0, entonces el resultado es 0
    if (yes == 0) or (no == 0):
        return 0

    entropy = (yes/(yes + no)) * math.log(yes/(yes + no), 2) + (no/(yes + no)) * math.log(no/(yes + no), 2)

    return entropy
"""
def entropia(conteo, categoria):
    llave = list(conteo.keys())[categoria]
    cantidades = conteo[llave].values()
    total = sum(cantidades)
    entropy = 0

    for cantidad in cantidades:
        if cantidad == 0:
            continue

        probabilidad = cantidad / total
        entropy += probabilidad * math.log(probabilidad, 2)

    return entropy


#prueba = contar_play(dataset, 0)
#print(entropia(prueba, 0))
#print(prueba[list(prueba.keys())[0]]["Yes"])
#print(prueba.items())

"""
def entropia_columna(values, columna):
    entropia_total = 0
    cantidad_datos = len(dataset)
    i = 0
    while i < len(values[columna]):
        llave = list(values[columna].keys())[i]
        entropy = (entropia(contar_play(dataset, columna), i) * (-1))
        entropia_total += entropy * (values[columna][llave] / cantidad_datos)

        i += 1

    return entropia_total
"""
def entropia_columna(data, columna):

    values = valores(data)
    entropia_total = 0
    cantidad_datos = len(data)
    i = 0

    while i < len(values[columna]):
        llave = list(values[columna].keys())[i]
        entropy = entropia(contar_play(data, columna), i) * (-1)
        entropia_total += entropy * (values[columna][llave] / cantidad_datos)
        i += 1

    return entropia_total

#print(entropia_columna(valores(dataset), 0))

# Esto calcula la entropia de la columna "Y" porque resulta que mi función de entropía no lo puede calcular (siempre devuelve 0).
# Podría modificar esta función para que recibiera el parámetro de colna y entonces reemplazara la función de entropia()
"""
def entropia_general(values):

    resultados = list(values[len(values)- 1].values())
    entropia = 0
    if resultados[0] == 0 or resultados[1] == 0:
        return entropia
    entropia = ((resultados[0] / (resultados[0] + resultados[1])) * math.log((resultados[0] / (resultados[0] + resultados[1])), 2) +
                (resultados[1] / (resultados[0] + resultados[1])) * math.log((resultados[1] / (resultados[0] + resultados[1])), 2))
    return entropia
"""
def entropia_general(values):

    resultados = list(values[len(values) - 1].values())
    total = sum(resultados)

    entropy = 0

    for cantidad in resultados:

        if cantidad == 0:
            continue

        probabilidad = cantidad / total
        entropy += probabilidad * math.log(probabilidad, 2)

    return entropy

#print(entropia_general(valores(dataset)))

# Calcula el information gain para una columna
def information_gain(data, columna):
    # Esto es solo para no tener que llamar repetidaente a valores()
    values = valores(data)
    general_entropy = (entropia_general(values) * (-1))
    column_entropy = entropia_columna(data, columna)
    info_gain = general_entropy - column_entropy

    return info_gain


#print(information_gain(dataset, 0))

#print(valores(dataset))
#print(entropia(contar_play(dataset, len(titulo_columnas)-1), 0))
#print(entropia(contar_play(dataset, 2), 0))

"""
def info_gain_list(data, descartadas):

    big_gains = 0
    columna_ganadora = 0
    i = 0
    while i < len(titulo_columnas) - 1:

        if i in descartadas:
            i += 1
        else:
            gain_columna = information_gain(data, i)
            if gain_columna > big_gains:
                big_gains = gain_columna
                columna_ganadora = i
            i += 1

    return columna_ganadora
"""

def info_gain_list(data, descartadas):

    big_gains = -1
    columna_ganadora = None
    i = 0

    while i < len(titulo_columnas) - 1:

        if i not in descartadas:
            gain_columna = information_gain(data, i)

            if gain_columna > big_gains:
                big_gains = gain_columna
                columna_ganadora = i

        i += 1

    return columna_ganadora

#print(info_gain_list(dataset, [0]))


def dividir_dataset(data, columna, valor):
    subconjunto = []
    i = 0
    while i < len(data):
        if data[i][columna] == valor:
            subconjunto.append(data[i])
        i += 1
    return subconjunto

#print(dividir_dataset(dataset, 0, "Sunny"))

"""
def es_puro(conjunto):
    yes = 0
    no = 0

    for fila in conjunto:
        if fila[-1] == "Yes":
            yes += 1
        else:
            no += 1

    if yes == 0 or no == 0:
        return True
    else:
        return False
"""

def es_puro(conjunto):

    clases = set()

    for fila in conjunto:
        clases.add(fila[-1])

    if len(clases) <= 1:
        return True
    else:
        return False

"""
def clase_mayoritaria(conjunto):
    yes = 0
    no = 0

    for fila in conjunto:
        if fila[-1] == "Yes":
            yes += 1
        else:
            no += 1

    if yes >= no:
        return "Yes"
    else:
        return "No"
"""

def clase_mayoritaria(conjunto):

    clases = Counter()

    for fila in conjunto:
        clases[fila[-1]] += 1

    return clases.most_common(1)[0][0]


"""
def construir_arbol(data, descartadas):

    # Si todos pertenecen a la misma clase, hemos llegado a una hoja
    if es_puro(data):
        return clase_mayoritaria(data)

    # Buscar la mejor columna disponible
    columna = info_gain_list(data, descartadas)

    # Crear el nodo
    arbol = {
        titulo_columnas[columna]: {}
    }

    # Obtener los valores posibles de esa columna
    valores_columna = valores(data)[columna].keys()

    # Crear una rama para cada valor
    for valor in valores_columna:

        subconjunto = dividir_dataset(data, columna, valor)

        arbol[titulo_columnas[columna]][valor] = construir_arbol(
            subconjunto,
            descartadas + [columna]
        )

    return arbol
"""

def construir_arbol(data, descartadas):

    # Si todos pertenecen a la misma clase, hemos llegado a una hoja
    if es_puro(data):
        return clase_mayoritaria(data)

    # Buscar la mejor columna disponible
    columna = info_gain_list(data, descartadas)

    # Si ya no existen columnas disponibles,
    # regresar la clase más común
    if columna is None:
        return clase_mayoritaria(data)

    # Crear el nodo
    arbol = {titulo_columnas[columna]: {}}

    # Obtener los valores posibles de esa columna
    valores_columna = valores(data)[columna].keys()

    # Crear una rama para cada valor
    for valor in valores_columna:
        subconjunto = dividir_dataset(data, columna, valor)
        arbol[titulo_columnas[columna]][valor] = construir_arbol(subconjunto, descartadas + [columna])

    return arbol

pprint.pprint(construir_arbol(dataset, []))

def predecir(arbol, dato):
    while isinstance(arbol, dict):
        columna = list(arbol.keys())[0]
        indice = titulo_columnas.index(columna)
        valor = dato[indice]
        arbol = arbol[columna][valor]
    return arbol

arbol = construir_arbol(dataset, [])
#nuevo_dato = ["Sunny", "Hot", "High", "Weak"]
nuevo_dato = ["vhigh", "vhigh", "2", "2", "small", "low"]

print(predecir(arbol, nuevo_dato))