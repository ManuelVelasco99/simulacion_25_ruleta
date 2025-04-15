import random
import matplotlib.pyplot as plt
import numpy as np
import sys

#PARAMETROS
cantidad_tiradas = int(sys.argv[2]) # -c cantidad_tiradas
cantidad_corridas = int(sys.argv[4]) # -n cantidad_corridas
numero_elegido = int(sys.argv[6]) # -e numero_elegido

# Generar datos de la corrida
def generar_corrida():
    numeros = range(0, 37)
    return [random.choice(numeros) for _ in range(cantidad_tiradas)]

# Generamos las corridas
corridas = []
for i in range(cantidad_corridas):
    corridas.append(generar_corrida())

def obtener_mediana(corrida):
    return np.median(corrida)

def obtener_varianza(corrida):
    return np.var(corrida)

def obtener_desviacion_estandar(corrida):
    return np.std(corrida)


# Gráfica freq relativa segun cantidad de tiradas
plt.figure(figsize=(10, 6))

for idx, corrida in enumerate(corridas):
    frecuencias_relativas = []
    tiradas = []
    contador = 0
    for i, numero in enumerate(corrida):
        if numero == numero_elegido:
            contador += 1
        tiradas.append(i + 1)
        frecuencia_relativa = contador / (i + 1)
        frecuencias_relativas.append(frecuencia_relativa)
    plt.plot(tiradas, frecuencias_relativas, alpha=0.9, label=f"Corrida {idx+1}")

plt.axhline(1/37, color='red', linestyle='--', label='Valor esperado (1/37)')
plt.xlabel("Cantidad de tiradas")
plt.ylabel("Frecuencia relativa")
plt.title(f"Frecuencia relativa del número {numero_elegido} en función de las tiradas de cada corrida")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Gráfica de desviación estándar en función de la cantidad de tiradas
plt.figure(figsize=(10, 6))

# Trazar una curva por cada corrida con la desviación estándar acumulada
for idx, corrida in enumerate(corridas):
    desviaciones_estandar = []
    for i in range(1, cantidad_tiradas + 1):
        muestra = corrida[:i]
        desviacion = np.std(muestra)
        desviaciones_estandar.append(desviacion)
    plt.plot(range(1, cantidad_tiradas + 1), desviaciones_estandar, alpha=0.9, label=f"Corrida {idx + 1}")

plt.xlabel("Cantidad de tiradas")
plt.ylabel("Desviación estándar del valor obtenido")
plt.title(f"Desviación estándar acumulada por corrida hasta cada tirada")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
