import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import math

valores_ruleta = {
    "rojo": [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36],
    "negro": [2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35],
    "impar": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35],
    "par": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36],
    "menor": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "mayor": [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
    "doc1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "doc2": [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    "doc3": [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
    "col1": [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34],
    "col2": [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
    "col3": [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]
}

capital_inicial = 10000
apuesta_inicial = 100
def generar_corrida():
    numeros = range(0, 37)
    return [random.choice(numeros) for _ in range(cantidad_tiradas)]

def generar_corridas():
    corridas = []
    for _ in range(cantidad_corridas):
        corridas.append(generar_corrida())
    return corridas

# PARAMETROS
cantidad_tiradas = int(sys.argv[2]) # -c cantidad_tiradas
cantidad_corridas = int(sys.argv[4]) # -n cantidad_corridas
numero_elegido = sys.argv[6] # -e numero_elegido
tipo_estrategia =  sys.argv[8]# -s tipo de estrategia
tipo_capital = sys.argv[10]# -a tipo de capital

valores_posibles_apuesta= [
    'rojo','negro','impar','par','mayor','menor','doc1','doc2','doc3','col1','col2','col3'
]
valores_posibles_apuesta.extend([str(i) for i in range(37)])
valores_posibles_estrategia = [
    'm','d','f','o'
]
valores_posibles_capital = [
    'm','d','f','o'
]

# Validación
if (cantidad_tiradas < 1):
    print("El número de tiradas debe ser mayor a 0.")
    exit(1)

if (cantidad_corridas < 1):
    print("El número de corridas debe ser mayor a 0.")
    exit(1)
if numero_elegido not in valores_posibles_apuesta:
    print("El valor ingresado debe ser un número entre 0 y 36 o alguno de los siguientes valores: 'rojo','negro','impar','par','mayor','menor','doc1','doc2','doc3','col1','col2','col3'")
    exit(1)
#@TODO agregar validaciones capital y estrategia

# Estrategia y capital

# print(f""" Simulación de una Ruleta  \n
#       "Número de Tiradas: {cantidad_tiradas} \n
#       "Número de Corridas: {cantidad_corridas} \n
#       "Número Elegido: {numero_elegido} \n
#       "Estrategia: {tipo_estrategia} \n
#       "Tipo de Capital: {tipo_capital} \n""")

# Generamos las corridas
corridas = generar_corridas()

def es_tirada_ganadora(tirada):
    if numero_elegido in ([str(i) for i in range(37)]):
        return tirada == int(numero_elegido)
    if tirada in valores_ruleta[numero_elegido]:
        return True
    else:
        return False
    

def girar_ruleta(indice_corrida,tirada):
    #@TODO
    #apuesta
    
    #si capital infinito return True
    #si capital bancarrota return False
    return

capital_corridas = []

for i,corrida in enumerate(corridas):
    capital = capital_inicial
    apuesta = apuesta_inicial
    for j,tirada in enumerate(corrida):
        #
        #if tipo_estrategia == 'm':
            #if j != 0 :
                #if es_tirada_ganadora(corrida[j-1]):


        gano = es_tirada_ganadora(tirada)
        print(tirada,gano,numero_elegido)
        #
        if not girar_ruleta(i,tirada,capital):
            break

