import random
import sys
import matplotlib.pyplot as plt # Importar matplotlib
import numpy as np # Importar numpy


class ContextoEstrategia:
    def __init__(self, patrimonio_actual, apuesta, tipo_capital):
        self.patrimonio_actual = patrimonio_actual
        self.apuesta = apuesta
        self.tipo_capital = tipo_capital

class Estrategia:
    def __init__(self, contexto):
        self.contexto = contexto

    def calcular_siguiente_apuesta(self, gano):
        raise NotImplementedError()

class MartingalaEstrategia(Estrategia):
    def __init__(self, contexto):
        super().__init__(contexto)
        self.apuesta_actual = contexto.apuesta

    def calcular_siguiente_apuesta(self, gano):
        if not gano:
            self.apuesta_actual *= 2
        else:
            self.apuesta_actual = apuesta

        if self.contexto.tipo_capital == "f" and self.apuesta_actual > self.contexto.patrimonio_actual:
            self.apuesta_actual = self.contexto.patrimonio_actual

        return self.apuesta_actual

class ParoliEstrategia(Estrategia):
    def __init__(self, contexto):
        super().__init__(contexto)
        self.apuesta_actual = contexto.apuesta

    def calcular_siguiente_apuesta(self, gano):
        if gano:
            self.apuesta_actual *= 2
        else:
            self.apuesta_actual = apuesta

        if self.contexto.tipo_capital == "f" and self.apuesta_actual > self.contexto.patrimonio_actual:
            self.apuesta_actual = self.contexto.patrimonio_actual

        return self.apuesta_actual

class DalamberEstrategia(Estrategia):
    def __init__(self, contexto):
        super().__init__(contexto)
        self.apuesta_actual = contexto.apuesta

    def calcular_siguiente_apuesta(self, gano):
        if not gano:
            self.apuesta_actual += apuesta
        elif self.apuesta_actual > apuesta:
            self.apuesta_actual -= apuesta

        if self.contexto.tipo_capital == "f" and self.apuesta_actual > self.contexto.patrimonio_actual:
            self.apuesta_actual = self.contexto.patrimonio_actual

        return self.apuesta_actual

class FibonacciEstrategia(Estrategia):
    def __init__(self, contexto):
        super().__init__(contexto)
        self.indice_fibonacci = 0
        self.secuencia_fibonacci = [1, 1]

    def calcular_siguiente_apuesta(self, gano):
        if not gano:
            self.indice_fibonacci += 1
            if self.indice_fibonacci >= len(self.secuencia_fibonacci):
                self.secuencia_fibonacci.append(
                    self.secuencia_fibonacci[-1] + self.secuencia_fibonacci[-2]
                )
        elif self.indice_fibonacci > 1:
            self.indice_fibonacci -= 2
        else:
            self.indice_fibonacci = 0

        apuesta_actual = self.secuencia_fibonacci[self.indice_fibonacci] * apuesta
        if self.contexto.tipo_capital == "f" and apuesta_actual > self.contexto.patrimonio_actual:
            apuesta_actual = self.contexto.patrimonio_actual

        return apuesta_actual

class Jugador:
    def __init__(self, eleccion, patrimonio_actual, tipo_estrategia, tipo_capital, apuesta):
        self.eleccion = eleccion
        self.patrimonio_actual = patrimonio_actual
        self.tipo_estrategia = tipo_estrategia
        self.tipo_capital = tipo_capital
        self.apuesta = apuesta
        self.estrategia = self.determinar_estrategia()

    def determinar_estrategia(self):
        contexto = ContextoEstrategia(self.patrimonio_actual, self.apuesta, tipo_capital)
        if self.tipo_estrategia.lower() == "m":
            return MartingalaEstrategia(contexto)
        elif self.tipo_estrategia.lower() == "f":
            return FibonacciEstrategia(contexto)
        elif self.tipo_estrategia.lower() == "d":
            return DalamberEstrategia(contexto)
        elif self.tipo_estrategia.lower() == "p":
            return ParoliEstrategia(contexto)
        return MartingalaEstrategia(contexto)

    def actualizar_estrategia(self):
        self.estrategia.contexto = ContextoEstrategia(self.patrimonio_actual, self.apuesta, tipo_capital)

class Ruleta:
    def __init__(self, jugador, cantidad_tiradas, cantidad_corridas):
        self.jugador = jugador
        self.cantidad_tiradas = cantidad_tiradas
        self.cantidad_corridas = cantidad_corridas
        self.generador_numeros = random.Random()
        self.historial_capital_corridas = []
        self.historial_apuestas_corridas = []
        self.historial_resultados_corridas = []  # Nuevo: para guardar si ganó o perdió cada tirada
        self.valores_por_eleccion = {
            "rojo": [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36],
            "negro": [2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35],
            "impar": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35],
            "par": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36],
            "menor": list(range(1, 19)),
            "mayor": list(range(19, 37)),
            "doc1": list(range(1, 13)),
            "doc2": list(range(13, 25)),
            "doc3": list(range(25, 37)),
            "col1": [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34],
            "col2": [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
            "col3": [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36],
        }

    def girar_ruleta(self):
        return self.generador_numeros.randint(0, 36)

    def puede_seguir_jugando(self):
        if self.jugador.tipo_capital == "i":
            return True

        return self.jugador.patrimonio_actual > 0 and self.jugador.apuesta <= self.jugador.patrimonio_actual

    def atino_numero(self, numero):
        if isinstance(self.jugador.eleccion, int):
            return numero == self.jugador.eleccion
        elif isinstance(self.jugador.eleccion, str):
            return numero in self.valores_por_eleccion.get(self.jugador.eleccion, [])
        return False

    def actualizar_patrimonio(self, gano):
        if not gano:
            self.jugador.patrimonio_actual -= self.jugador.apuesta
        else:
            if isinstance(self.jugador.eleccion, int):
                pago = self.jugador.apuesta * 35
            elif self.jugador.eleccion in ["rojo", "negro", "par", "impar", "mayor", "menor"]:
                pago = self.jugador.apuesta
            elif self.jugador.eleccion.startswith("doc") or self.jugador.eleccion.startswith("col"):
                pago = self.jugador.apuesta * 2
            else:
                raise ValueError("Tipo de elección no soportado")
            self.jugador.patrimonio_actual += pago + self.jugador.apuesta

    def empezar_juego(self):
        patrimonio_inicial = self.jugador.patrimonio_actual
        apuesta_inicial = self.jugador.apuesta
        self.historial_apuestas_corridas = [] # Limpiar historial al empezar
        self.historial_capital_corridas = [] # Limpiar historial de capital al empezar
        self.historial_resultados_corridas = []

        for i, corrida in enumerate(range(self.cantidad_corridas)):
            self.jugador.patrimonio_actual = patrimonio_inicial
            self.jugador.apuesta = apuesta_inicial
            # Reiniciar estrategia para cada corrida (importante para Fibonacci/Dalamber)
            self.jugador.estrategia = self.jugador.determinar_estrategia()

            historial_apuestas_corrida_actual = [] # Historial para esta corrida
            historial_capital_corrida_actual = [patrimonio_inicial] # Empezar con capital inicial
            historial_resultados_corrida_actual = []  # Nuevo: Historial de 1 (ganó) o 0 (perdió)

            for tirada_num in range(self.cantidad_tiradas):
                # Verificar si se puede seguir jugando (capital finito)
                if not self.puede_seguir_jugando():
                    # Si no puede seguir, rellenar el resto de tiradas con el capital actual (0 o lo que quede)
                    capital_final_corrida = self.jugador.patrimonio_actual # Capital before this impossible tirada
                    num_tiradas_restantes = self.cantidad_tiradas - tirada_num
                    # Pad bets with 0 for remaining tiradas
                    historial_apuestas_corrida_actual.extend([0] * num_tiradas_restantes)
                    # Pad capital history with the final capital value for the remaining tiradas
                    historial_capital_corrida_actual.extend([capital_final_corrida] * num_tiradas_restantes)
                    # Pad results history with 0 (loss/no play) for remaining tiradas
                    historial_resultados_corrida_actual.extend([0] * num_tiradas_restantes)
                    print(f"  Corrida {i} detenida en tirada {tirada_num} por falta de capital o apuesta imposible.")
                    break # Salir del bucle de tiradas para esta corrida

                apuesta_esta_tirada = self.jugador.apuesta
                historial_apuestas_corrida_actual.append(apuesta_esta_tirada)

                numero = self.girar_ruleta()
                gano = self.atino_numero(numero)
                self.actualizar_patrimonio(gano)
                historial_capital_corrida_actual.append(self.jugador.patrimonio_actual) # Guardar capital *después* de actualizar

                # Actualizar contexto de la estrategia antes de calcular la siguiente apuesta
                self.jugador.actualizar_estrategia()
                self.jugador.apuesta = self.jugador.estrategia.calcular_siguiente_apuesta(gano)
                historial_resultados_corrida_actual.append(int(gano))  # Guardar 1 si ganó, 0 si perdió

            # Asegurarse de que los historiales tengan la longitud correcta si la corrida terminó antes
            while len(historial_capital_corrida_actual) < self.cantidad_tiradas + 1:
                 historial_capital_corrida_actual.append(historial_capital_corrida_actual[-1])
            while len(historial_apuestas_corrida_actual) < self.cantidad_tiradas:
                 historial_apuestas_corrida_actual.append(0)
            while len(historial_resultados_corrida_actual) < self.cantidad_tiradas:
                historial_resultados_corrida_actual.append(0)

            self.historial_apuestas_corridas.append(historial_apuestas_corrida_actual)
            self.historial_capital_corridas.append(historial_capital_corrida_actual)
            self.historial_resultados_corridas.append(historial_resultados_corrida_actual)

            print(f"Corrida {i} terminada. Patrimonio final: {self.jugador.patrimonio_actual}")

def graficar_histograma(resultados):
    promedio_general_corridas = []
    for i, corrida in enumerate(resultados):
        tiradas_eje_x = np.arange(1, len(corrida) + 1)
        frecuencia_relativa = np.cumsum(corrida) / tiradas_eje_x
        promedio_general_corridas.append(frecuencia_relativa)

    promedio_general_corridas_nuevo = np.mean(promedio_general_corridas, axis=0) # Promedio de todas las corridas
    plt.figure(figsize=(10, 6))
    plt.bar(tiradas_eje_x, promedio_general_corridas_nuevo, color='red', edgecolor='red', width=0.8)
    plt.xlabel('n (número de tiradas)')
    plt.ylabel('fr (frecuencia relativa)')
    plt.title(f'frsa (Frecuencia Relativa de Aciertos según n)\nCorrida- Estrategia: {tipo_estrategia.capitalize()}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    primera_corrida = resultados[0]
    tiradas_eje_x = np.arange(1, len(primera_corrida) + 1)
    frecuencia_relativa = np.cumsum(primera_corrida) / tiradas_eje_x
    plt.figure(figsize=(10, 6))
    plt.bar(tiradas_eje_x, frecuencia_relativa, color='red', edgecolor='red', width=0.8)
    plt.xlabel('n (número de tiradas)')
    plt.ylabel('fr (frecuencia relativa)')
    plt.title(f'frsa (Frecuencia Relativa de Aciertos según n)\nCorrida- Estrategia: {tipo_estrategia.capitalize()}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def graficar_evolucion_capital(esResultadoPromedio):
    plt.figure(figsize=(12, 7))

    tiradas_eje_x = np.arange(ruleta.cantidad_tiradas + 1)

    # Graficar línea de capital inicial como referencia
    capital_inicial_linea = np.full_like(tiradas_eje_x, fill_value=capital_inicial, dtype=float)
    plt.plot(tiradas_eje_x, capital_inicial_linea, color='blue', linestyle='--', linewidth=1.5, label='Capital Inicial')

    if(esResultadoPromedio):
        # Calcular el promedio de capital a lo largo de las corridas
        promedio_capital = np.mean(ruleta.historial_capital_corridas, axis=0)
        plt.plot(tiradas_eje_x, promedio_capital, marker='', linestyle='-', linewidth=2, color='orange', label='Promedio Capital')
        plt.title(f'Evolución del Capital promedio de las corridas - Estrategia: {tipo_estrategia.capitalize()}')
    else:
        # Graficar la evolución del capital para cada corrida
        for i, capital_historial in enumerate(ruleta.historial_capital_corridas):
            # Asegurarse que el historial tenga la longitud correcta para graficar
            if len(capital_historial) == len(tiradas_eje_x):
                plt.plot(tiradas_eje_x, capital_historial, marker='', linestyle='-', linewidth=1, alpha=0.8, label=f'Corrida {i+1}')
            else:
                print(f"Advertencia: La corrida {i+1} tiene una longitud de historial de capital inesperada ({len(capital_historial)}) y no se graficará.")

            plt.title(f'Evolución del Capital por Corrida ({ruleta.cantidad_corridas} Corridas - Estrategia: {tipo_estrategia.capitalize()})')

    plt.xlabel('Número de Tirada (n)')
    plt.ylabel('Capital (cc)')
    plt.grid(True, linestyle='--', alpha=0.5)
    # Mostrar leyenda solo si hay pocas corridas para no saturar
    if ruleta.cantidad_corridas <= 5:
        plt.legend()
    plt.tight_layout()
    plt.show()

### Ejecución

# PARAMETROS
apuesta = 250
capital_inicial = 1_000

cantidad_tiradas = int(sys.argv[2]) # -c cantidad_tiradas
cantidad_corridas = int(sys.argv[4]) # -n cantidad_corridas
eleccion = sys.argv[6] # -e numero_elegido
tipo_estrategia = sys.argv[8] # -s estrategia
tipo_capital = sys.argv[10] # -a tipo_capital

if cantidad_tiradas < 1:
    print("El número de tiradas debe ser mayor a 0.")
    sys.exit(1)

if cantidad_corridas < 1:
    print("El número de corridas debe ser mayor a 0.")
    sys.exit(1)

valid_string_choices = ["rojo", "negro", "impar", "par", "menor", "mayor", "doc1", "doc2", "doc3", "col1", "col2", "col3"]
is_valid_choice = False
try:
    # Intentar convertir a entero
    choice_int = int(eleccion)
    if 0 <= choice_int <= 36:
        eleccion = choice_int # Guardar como entero si es válido
    else:
        print(f"Error: La elección numérica '{eleccion}' debe estar entre 0 y 36.")
        sys.exit(1)
except ValueError:
    # Si no es entero, verificar si es una cadena válida
    if eleccion.lower() in valid_string_choices:
        eleccion = eleccion.lower() # Guardar en minúsculas
    else:
        print(f"Error: La elección '{eleccion}' no es válida. Opciones: 0-36 o {', '.join(valid_string_choices)}.")
        sys.exit(1)

valid_strategies = ["m", "f", "d", "p"]
if tipo_estrategia.lower() not in valid_strategies:
    print(f"Error: Estrategia '{tipo_estrategia}' no válida. Opciones: {', '.join(valid_strategies)}.")
    sys.exit(1)
tipo_estrategia = tipo_estrategia.lower()

# Validar tipo_capital
valid_capital_types = ["f", "i"]
if tipo_capital.lower() not in valid_capital_types:
    print(f"Error: Tipo de capital '{tipo_capital}' no válido. Opciones: 'f' (finito) o 'i' (infinito).")
    sys.exit(1)
tipo_capital = tipo_capital.lower()

if apuesta <= 0:
    print("Error: La apuesta inicial debe ser mayor a 0.")
    sys.exit(1)

# Validar capital inicial (si es finito)
if tipo_capital == 'f' and capital_inicial <= 0:
    print("Error: El capital inicial debe ser mayor a 0 para capital finito.")
    sys.exit(1)
if tipo_capital == 'f' and apuesta > capital_inicial:
    print("Error: La apuesta inicial no puede ser mayor que el capital inicial para capital finito.")
    sys.exit(1)

jugador = Jugador(eleccion, capital_inicial, tipo_estrategia, tipo_capital, apuesta)
ruleta = Ruleta(jugador, cantidad_tiradas, cantidad_corridas)
ruleta.empezar_juego()


graficar_evolucion_capital(False) # Graficar la evolución del capital por corrida
graficar_evolucion_capital(True) # Graficar el promedio de capital por corrida
graficar_histograma(ruleta.historial_resultados_corridas) # Graficar el histograma de resultados