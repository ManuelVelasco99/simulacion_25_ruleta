import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import math

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
todas_las_desviaciones_estandar = []
for idx, corrida in enumerate(corridas):
    desviaciones_estandar = []
    for i in range(1, cantidad_tiradas + 1):
        muestra = corrida[:i]
        desviacion = obtener_desviacion_estandar(muestra)
        desviaciones_estandar.append(desviacion)
    todas_las_desviaciones_estandar.append(desviaciones_estandar)
    plt.plot(range(1, cantidad_tiradas + 1), desviaciones_estandar, alpha=0.9, label=f"Corrida {idx + 1}")

plt.xlabel("Cantidad de tiradas")
plt.ylabel("Desviación estándar del valor obtenido")
plt.title(f"Desviación estándar acumulada por corrida hasta cada tirada")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Calcular la desviación estándar promedio en función de las tiradas
desviacion_estandar_promedio = np.mean(todas_las_desviaciones_estandar, axis=0)

# Gráfica de la desviación estándar promedio en función de la cantidad de tiradas
plt.figure(figsize=(10, 6))
plt.plot(range(1, cantidad_tiradas + 1), desviacion_estandar_promedio, label="Desviación estándar promedio")

# Valor esperado de la desviación estandar para una distribución uniforme de 0 a 36
valor_esperado_desviacion_estandar = math.sqrt(((37**2) - 1) / 12)
plt.axhline(valor_esperado_desviacion_estandar, color='red', linestyle='--', label=f'Valor esperado ({valor_esperado_desviacion_estandar:.2f})')

plt.xlabel("Cantidad de tiradas")
plt.ylabel("Desviación estándar promedio del valor obtenido")
plt.title("Desviación estándar promedio acumulada hasta cada tirada")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Gráfica de varianza en función de la cantidad de tiradas
plt.figure(figsize=(10, 6))

# Trazar una curva por cada corrida con la varianza acumulada
varianzas_de_las_corridas = []
for idx, corrida in enumerate(corridas):
    varianzas_de_la_corrida = []
    for i in range(1, cantidad_tiradas + 1):
        muestra = corrida[:i]
        varianza = obtener_varianza(muestra) if i > 1 else 0
        varianzas_de_la_corrida.append(varianza)
    varianzas_de_las_corridas.append(varianzas_de_la_corrida)
    plt.plot(range(1, cantidad_tiradas + 1), varianzas_de_la_corrida, alpha=0.9, label=f"Corrida {idx + 1}")

plt.xlabel("Cantidad de tiradas")
plt.ylabel("Varianza del valor obtenido")
plt.title(f"Varianza acumulada por corrida hasta cada tirada")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Calcular la varianza promedio en función de las tiradas
varianzas_promedio = np.mean(varianzas_de_las_corridas, axis=0)

# Gráfica de varianza promedio en función de la cantidad de tiradas
plt.figure(figsize=(10, 6))
plt.plot(range(1, cantidad_tiradas + 1), varianzas_promedio, label="Varianza promedio")

# Valor esperado de la varianza para una distribución uniforme de 0 a 36
valor_esperado_varianza = ((37**2) - 1) / 12
plt.axhline(valor_esperado_varianza, color='red', linestyle='--', label=f'Valor esperado ({valor_esperado_varianza:.2f})')

plt.xlabel("Cantidad de tiradas")
plt.ylabel("Varianza promedio del valor obtenido")
plt.title("Varianza promedio acumulada hasta cada tirada")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()