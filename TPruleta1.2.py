import random

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

        if self.contexto.tipo_capital == "finito" and self.apuesta_actual > self.contexto.patrimonio_actual:
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

        if self.contexto.tipo_capital == "finito" and self.apuesta_actual > self.contexto.patrimonio_actual:
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

        if self.contexto.tipo_capital == "finito" and self.apuesta_actual > self.contexto.patrimonio_actual:
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

        print(f"Indice Fibonacci: {self.indice_fibonacci}")
        print(f"Secuencia Fibonacci: {self.secuencia_fibonacci}")
        apuesta_actual = self.secuencia_fibonacci[self.indice_fibonacci] * apuesta
        if self.contexto.tipo_capital == "finito" and apuesta_actual > self.contexto.patrimonio_actual:
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
        if self.tipo_estrategia.lower() == "martingala":
            return MartingalaEstrategia(contexto)
        elif self.tipo_estrategia.lower() == "fibonacci":
            return FibonacciEstrategia(contexto)
        elif self.tipo_estrategia.lower() == "dalamber":
            return DalamberEstrategia(contexto)
        elif self.tipo_estrategia.lower() == "paroli":
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
        if self.jugador.tipo_capital == "infinito":
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
        for i, corrida in enumerate(range(self.cantidad_corridas)):
            self.jugador.patrimonio_actual = patrimonio_inicial
            self.jugador.apuesta = apuesta_inicial
            for _ in range(self.cantidad_tiradas):
                numero = self.girar_ruleta()
                gano = self.atino_numero(numero)
                self.actualizar_patrimonio(gano)
                self.jugador.actualizar_estrategia()
                print(f"Apuesta: {self.jugador.apuesta}")
                self.jugador.apuesta = self.jugador.estrategia.calcular_siguiente_apuesta(gano)
                print(f"Patrimonio: {self.jugador.patrimonio_actual}, Gano: {gano}")
                if not self.puede_seguir_jugando():
                    break

cantidad_tiradas = 10
cantidad_corridas = 1
eleccion = "mayor"
tipo_estrategia = "Fibonacci"
tipo_capital = "infinito"
apuesta = 1000
jugador = Jugador(eleccion, 100_000, tipo_estrategia, tipo_capital, apuesta)
ruleta = Ruleta(jugador, cantidad_tiradas, cantidad_corridas)
ruleta.empezar_juego()
