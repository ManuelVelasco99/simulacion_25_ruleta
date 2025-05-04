import matplotlib.pyplot as plt
import numpy as np
import random
from math import log2
import pandas as pd
from scipy import stats

# Prueba de distribución uniforme
def prueba_distribucion_uniforme(numeros, nombre_generador):
    plt.hist(numeros, bins=50, color='blue', alpha=0.7)
    plt.title(f'Distribución Uniforme - {nombre_generador}')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    plt.savefig(f'./images/TP2.1/distribucion_{nombre_generador}.png')
    plt.show()

# Prueba de media y varianza
def prueba_media_varianza(numeros, nombre_generador):
    media = np.mean(numeros)
    varianza = np.var(numeros)
    print('----------------------')
    print(f'Media de los números generados por {nombre_generador}: {media}')
    print(f'Varianza de los números generados por {nombre_generador}: {varianza}')


def prueba_chi_cuadrado(numeros, nombre_generador):
    # Esperamos que haya una frecuencia esperada en cada bin
    subintervalos =  int(1 + log2(len(numeros))) # para 512^2 = 262144
    frec_esperada = len(numeros) / subintervalos
    # Generar histograma y obtener la frecuencia observada
    frec_observada, _, _ = plt.hist(numeros, bins=subintervalos, color='blue', alpha=0.7, edgecolor='black')

    # Calcular chi-cuadrado
    chi_cuadrado, p_valor = stats.chisquare(frec_observada, [frec_esperada]*len(frec_observada))
    print(f'Chi-cuadrado para {nombre_generador}: {chi_cuadrado}')
    print(f'Frecuencia observada para {nombre_generador}: {sum(frec_observada)/len(frec_observada)} vs {frec_esperada}')
    print(f'p-valor para {nombre_generador}: {p_valor}')

    # Añadir títulos y etiquetas
    plt.title(f'Histograma de {nombre_generador}')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')

    # Guardar la gráfica
    plt.savefig(f'./images/TP2.1/histograma_{nombre_generador}.png')
    plt.show()

    return chi_cuadrado, p_valor

# Implementación del Generador Congruencial Lineal (GCL)
def generador_gcl(m, a, c, seed, n):
    numeros = []
    x = seed
    for _ in range(n):
        x = (a * x + c) % m
        numeros.append(x)
    return numeros

# Implementación del Generador de Mersenne Twister
def generador_mersenne_twister(seed, n):
    random.seed(seed)
    return [random.getrandbits(32) for _ in range(n)]

def crear_imagen_con_ruido(numbers, size,nombre_generador):
    # Normaliza los números para que estén en el rango [0, 1]
    normalized_numbers = np.array(numbers) / max(numbers)
    # Devuelve una matriz cuadrada del tamaño de size
    noise_image = normalized_numbers.reshape(size, size)

    plt.imshow(noise_image, cmap='gray')
    plt.title(f'Ruido atmosferico - {nombre_generador}')
    plt.axis('off')
    plt.savefig(f'./images/TP2.1/Ruido atmosferico {nombre_generador}.png')
    plt.show()

def generar_graficas():
    numbers = generador_gcl(modulo, multiplicador, incremento, seed, numeros_a_generar)
    numeros_mt = generador_mersenne_twister(seed, numeros_a_generar)
    numeros_python = [random.randint(0, 2**32 - 1) for _ in range(numeros_a_generar)]

    for numeros, nombre_generador in [
            (numbers, 'GCL'),
            (numeros_mt, 'Mersenne Twister'),
            (numeros_python, 'Python Random')]:

        crear_imagen_con_ruido(numeros,tamaño,nombre_generador)

    prueba_chi_cuadrado(numbers, 'GCL')
    prueba_chi_cuadrado(numeros_mt, 'Mersenne Twister')
    prueba_chi_cuadrado(numeros_python, 'Python Random')

    prueba_distribucion_uniforme(numbers, 'GCL')
    prueba_distribucion_uniforme(numeros_mt, 'Mersenne Twister')
    prueba_distribucion_uniforme(numeros_python, 'Python Random')

    prueba_media_varianza(numbers, 'GCL')
    prueba_media_varianza(numeros_mt, 'Mersenne Twister')
    prueba_media_varianza(numeros_python, 'Python Random')

# Parámetros
modulo = 2**32 # Módulo
multiplicador = 1664525 # Multiplicador
incremento = 16457893108 # Incremento
seed = 12345 # Semilla
tamaño = 512 # Tamaño de la imagen
numeros_a_generar = tamaño ** 2 # Cantidad de números a generar

generar_graficas()