import random
import sys
import matplotlib.pyplot as plt  # Importar matplotlib
import pandas as pd
import numpy as np  # Importar numpy

estrategias_apuesta = {
    'm': 'Martingala',
    'f': 'Fibonacci',
    'd': "D'Alembert",
    'p': 'Paroli'
}

capital = {
    'i': 'infinito',
    'f': 'finito',
}

class Estrategia:
    def calcular_siguiente_apuesta(self, capital, apuesta_anterior, gano, cantidad_victorias_seguidas):
        raise NotImplementedError()


class MartingalaEstrategia(Estrategia):
    def calcular_siguiente_apuesta(self, capital, apuesta_anterior, gano, cantidad_victorias_seguidas):
        apuesta = apuesta_anterior
        if not gano:
            apuesta = apuesta * 2
        else:
            apuesta = apuesta_inicial

        if tipo_capital == "f" and apuesta > capital:
            apuesta = capital

        return apuesta


class ParoliEstrategia(Estrategia):
    def calcular_siguiente_apuesta(self, capital, apuesta_anterior, gano, cantidad_victorias_seguidas):
        apuesta = apuesta_anterior
        if gano and cantidad_victorias_seguidas < 3:
            apuesta = apuesta * 2
        else:
            apuesta = apuesta_inicial

        if tipo_capital == "f" and apuesta > capital:
            apuesta = capital

        return apuesta


class DalembertEstrategia(Estrategia):
    def calcular_siguiente_apuesta(self, capital, apuesta_anterior, gano, cantidad_victorias_seguidas):
        apuesta = apuesta_anterior
        if gano:
            apuesta = max(apuesta - apuesta_inicial, apuesta_inicial)
        else:
            apuesta += apuesta_inicial

        if tipo_capital == "f" and apuesta > capital:
            apuesta = capital

        return apuesta


class FibonacciEstrategia(Estrategia):
    def __init__(self):
        self.indice_fibonacci = 0
        self.secuencia_fibonacci = [1, 1]

    def calcular_siguiente_apuesta(self, capital, apuesta_anterior, gano, cantidad_victorias_seguidas):
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

        apuesta = self.secuencia_fibonacci[self.indice_fibonacci] * apuesta_inicial

        if tipo_capital == "f" and apuesta > capital:
            apuesta = capital

        return apuesta


class Jugador:
    def __init__(self):
        self.apuesta = apuesta_inicial
        self.capital = capital_inicial


class Ruleta:
    def __init__(self, jugador):
        self.jugador = jugador
        self.historial_capital_corridas = []
        self.historial_apuestas_corridas = []
        self.historial_resultados_corridas = []  # Nuevo: para guardar si ganó o perdió cada tirada
        self.historial_frecuencia_relativa = []
        self.historial_dfs = []
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

    def determinar_estrategia(self):
        if tipo_estrategia.lower() == "m":
            return MartingalaEstrategia()
        elif tipo_estrategia.lower() == "f":
            return FibonacciEstrategia()
        elif tipo_estrategia.lower() == "d":
            return DalembertEstrategia()
        elif tipo_estrategia.lower() == "p":
            return ParoliEstrategia()
        return MartingalaEstrategia()

    def atino_numero(self, numero):
        if isinstance(eleccion, int):
            return numero == eleccion
        elif isinstance(eleccion, str):
            return numero in self.valores_por_eleccion.get(eleccion, [])
        return False

    def actualizar_patrimonio(self, gano):
        if not gano:
            self.jugador.patrimonio_actual -= self.jugador.apuesta
        else:
            if isinstance(eleccion, int):
                pago = self.jugador.apuesta * 35
            elif eleccion in ["rojo", "negro", "par", "impar", "mayor", "menor"]:
                pago = self.jugador.apuesta
            elif eleccion.startswith("doc") or eleccion.startswith("col"):
                pago = self.jugador.apuesta * 2
            else:
                raise ValueError("Tipo de elección no soportado")
            self.jugador.patrimonio_actual += pago

    def empezar_juego(self):
        self.historial_apuestas_corridas = []  # Limpiar historial al empezar
        self.historial_capital_corridas = []  # Limpiar historial de capital al finalizar
        self.historial_resultados_corridas = []
        self.historial_frecuencia_relativa = []
        self.historial_dfs = []

        for i, corrida in enumerate(range(cantidad_corridas)):

            tiradas = np.random.randint(0, 37, size=[cantidad_tiradas])
            self.jugador.patrimonio_actual = capital_inicial
            self.jugador.apuesta = apuesta_inicial

            estrategia = self.determinar_estrategia()

            historial_apuestas_corrida_actual = []  # Historial para esta corrida
            historial_capital_corrida_actual = []  # Empezar con capital inicial
            historial_resultados_corrida_actual = []  # Nuevo: Historial de 1 (ganó) o 0 (perdió)
            historial_frecuencia_relativa_actual = []

            historial_apuestas_corrida_actual.append(self.jugador.apuesta)
            historial_capital_corrida_actual.append(self.jugador.patrimonio_actual)

            cantidad_victorias_seguidas = 0

            for index, tirada in enumerate(tiradas):

                is_win = self.atino_numero(tirada)

                if cantidad_victorias_seguidas == 3:
                    cantidad_victorias_seguidas = 0

                if is_win:
                    cantidad_victorias_seguidas += 1
                else:
                    cantidad_victorias_seguidas = 0

                historial_resultados_corrida_actual.append(is_win)

                historial_frecuencia_relativa_actual.append(
                    historial_resultados_corrida_actual.count(True) / (index + 1))

                self.actualizar_patrimonio(is_win)

                self.jugador.apuesta = estrategia.calcular_siguiente_apuesta(self.jugador.capital, self.jugador.apuesta,
                                                                             is_win, cantidad_victorias_seguidas)

                if self.jugador.apuesta > self.jugador.patrimonio_actual and tipo_capital == 'f':  # banca rota
                    completar_array(historial_capital_corrida_actual,0,cantidad_tiradas)
                    completar_array(historial_apuestas_corrida_actual,0,cantidad_tiradas)
                    completar_array(historial_resultados_corrida_actual,False,cantidad_tiradas)
                    completar_array(historial_frecuencia_relativa_actual,0,cantidad_tiradas)
                    break

                if len(historial_apuestas_corrida_actual) == cantidad_tiradas:
                    break

                historial_apuestas_corrida_actual.append(self.jugador.apuesta)
                historial_capital_corrida_actual.append(self.jugador.patrimonio_actual)

            self.historial_apuestas_corridas.append(historial_apuestas_corrida_actual)
            self.historial_capital_corridas.append(historial_capital_corrida_actual)
            self.historial_resultados_corridas.append(historial_resultados_corrida_actual)
            self.historial_frecuencia_relativa.append(historial_frecuencia_relativa_actual)

            dataframe = pd.DataFrame({
                'capital': historial_capital_corrida_actual,
                'apuesta': historial_apuestas_corrida_actual,
                'victoria': historial_resultados_corrida_actual,
                'fr': historial_frecuencia_relativa_actual
            })

            self.historial_dfs.append(dataframe)

            print(f"Corrida {i+1} terminada. Patrimonio final: {self.jugador.patrimonio_actual}")

    def graficar(self):
        # Mostrar gráfico
        self.grafico_flujo_caja()
        self.graficar_histograma()

        listados_capital = [df['capital'] for df in self.historial_dfs]
        listado_frecuencias_relativas = [df["fr"] for df in self.historial_dfs]

        self.grafico_flujo_caja_promedio(listados_capital)
        self.generar_histograma_promedio(listado_frecuencias_relativas)

    def grafico_flujo_caja(self):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.set_title(f"Evolución del capital promedio de las tiradas - Estrategia: {estrategias_apuesta[tipo_estrategia]} - Tipo de capital: {capital[tipo_capital]}")
        ax.set_xlabel('n (número de tiradas)')
        ax.set_ylabel('c (capital)')

        print(self.historial_capital_corridas)

        for capital_tiradas in self.historial_capital_corridas:
            ax.plot(capital_tiradas, linewidth=2.0, )
            max_value = max(capital_tiradas)
            ax.scatter([0], [max_value])

        ax.axhline(capital_inicial, color='r', linestyle='--', label='fci (flujo de caja inicial)')

        ax.ticklabel_format(axis='y', style='plain')

        plt.legend()
        plt.show()

    def graficar_histograma(self):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.set_title(f"Histograma para {cantidad_corridas} corridas - Estrategia: {estrategias_apuesta[tipo_estrategia]} - Tipo de capital: {capital[tipo_capital]}")
        ax.set_xlabel('n (número de tiradas)')
        ax.set_ylabel('fr (frecuencia relativa)')

        for df in self.historial_dfs:
            ax.bar(df.index, df['fr'], alpha=0.5)

        ax.axhline(self.obtener_valor_esperado_frecuancia_relativa(), color='r', linestyle='--', label='Valor esperado')
        plt.legend()
        plt.show()

    def grafico_flujo_caja_promedio(self, listados_capital):
        fig, ax = plt.subplots(figsize=(10, 6))

        max_tiradas = max(len(listado) for listado in listados_capital)
        promedio_capital = []
        for j in range(0, max_tiradas):
            sum_capital_en_tiradas_i = 0
            for i in range(0, len(listados_capital)):
                try:
                    sum_capital_en_tiradas_i += listados_capital[i][j]
                except:
                    sum_capital_en_tiradas_i += 0
            promedio_capital.append(sum_capital_en_tiradas_i / cantidad_corridas)

        # Graficar el promedio del capital
        ax.plot(promedio_capital, label='Promedio del Capital', color='blue')
        ax.set_title(f'Flujo de caja promedio a lo largo de las tiradas para {cantidad_corridas} corridas - Estrategia: {estrategias_apuesta[tipo_estrategia]} - Tipo de capital: {capital[tipo_capital]}')
        ax.set_xlabel('Número de tiradas')
        ax.set_ylabel('Capital Promedio')
        ax.legend()
        ax.grid(True)
        plt.show()

    def obtener_valor_esperado_frecuancia_relativa(self):
        cantidad_numeros_ganadores = len(self.valores_por_eleccion[eleccion])
        return cantidad_numeros_ganadores / 37

    def generar_histograma_promedio(self, listado_frecuencia_relativa):
        fig, ax = plt.subplots(figsize=(10, 6))

        max_tiradas = max(len(listado) for listado in listado_frecuencia_relativa)
        promedio_frecuencias = []

        for j in range(max_tiradas):
            sum_frecuencias_en_tirada_j = 0
            for i in range(len(listado_frecuencia_relativa)):
                try:
                    sum_frecuencias_en_tirada_j += listado_frecuencia_relativa[i][j]
                except:
                    sum_frecuencias_en_tirada_j += 0
            promedio_frecuencias.append(sum_frecuencias_en_tirada_j / cantidad_corridas)

        x_values = range(len(promedio_frecuencias))

        ax.set_title(f"Histograma promedio de {cantidad_corridas} corridas - Estrategia: {estrategias_apuesta[tipo_estrategia]} - Tipo de capital: {capital[tipo_capital]}")
        ax.set_xlabel('n (número de tiradas)')
        ax.set_ylabel('fr (frecuencia relativa)')
        ax.bar(x_values, promedio_frecuencias, alpha=0.5)
        
        ax.axhline(self.obtener_valor_esperado_frecuancia_relativa(), color='r', linestyle='--', label='Valor esperado')
        plt.legend()
        plt.show()


def completar_array(array, valor, longitud_objetivo):
    longitud_actual = len(array)
    if longitud_actual >= longitud_objetivo:
        return array[:longitud_objetivo]  # Recortar si excede
    cantidad_a_agregar = longitud_objetivo - longitud_actual
    array.extend([valor] * cantidad_a_agregar)
    return array

# Ejecución

# Parámetros
apuesta_inicial = 100
capital_inicial = 10000

cantidad_tiradas = int(sys.argv[2])  # -c cantidad_tiradas
cantidad_corridas = int(sys.argv[4])  # -n cantidad_corridas
eleccion = sys.argv[6]  # -e numero_elegido
tipo_estrategia = sys.argv[8]  # -s estrategia
tipo_capital = sys.argv[10]  # -a tipo_capital

if cantidad_tiradas < 1:
    print("El número de tiradas debe ser mayor a 0.")
    sys.exit(1)

if cantidad_corridas < 1:
    print("El número de corridas debe ser mayor a 0.")
    sys.exit(1)

valid_string_choices = ["rojo", "negro", "impar", "par", "menor", "mayor", "doc1", "doc2", "doc3", "col1", "col2",
                        "col3"]
is_valid_choice = False
try:
    # Intentar convertir a entero
    choice_int = int(eleccion)
    if 0 <= choice_int <= 36:
        eleccion = choice_int  # Guardar como entero si es válido
    else:
        print(f"Error: La elección numérica '{eleccion}' debe estar entre 0 y 36.")
        sys.exit(1)
except ValueError:
    # Si no es entero, verificar si es una cadena válida
    if eleccion.lower() in valid_string_choices:
        eleccion = eleccion.lower()  # Guardar en minúsculas
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

if apuesta_inicial <= 0:
    print("Error: La apuesta inicial debe ser mayor a 0.")
    sys.exit(1)

# Validar capital inicial (si es finito)
if tipo_capital == 'f' and capital_inicial <= 0:
    print("Error: El capital inicial debe ser mayor a 0 para capital finito.")
    sys.exit(1)
if tipo_capital == 'f' and apuesta_inicial > capital_inicial:
    print("Error: La apuesta inicial no puede ser mayor que el capital inicial para capital finito.")
    sys.exit(1)

jugador = Jugador()
ruleta = Ruleta(jugador)
ruleta.empezar_juego()
ruleta.graficar()
