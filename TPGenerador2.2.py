import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import binom, expon,chisquare, gamma, hypergeom, norm, nbinom, poisson, randint

def test_chi_cuadrado_binomial(muestras):
  # Calcular frecuencias observadas
  observed_freq = np.bincount(muestras, minlength=n+1)

  # Calcular frecuencias esperadas
  expected_freq = np.array([binom.pmf(k, n, p) * len(muestras) for k in range(n+1)])

  # Realizar la prueba chi-cuadrado
  chi2_stat, p_value = chisquare(observed_freq, expected_freq)

  print(f"Chi-cuadrado: {chi2_stat}")
  print(f"p-valor: {p_value}")

  # Interpretación
  alpha = 0.05
  if p_value < alpha:
      print("No supera el test")
  else:
      print("Supera el test")

def graficar_binomial(n, p, size):
    # Generar muestras
    samples = np.random.binomial(n=n, p=p, size=size)
    # Histograma
    plt.figure(figsize=(8, 6))
    sns.histplot(samples, bins=30, kde=False, color='skyblue')
    plt.title('Histograma de muestras binomiales (n=100, p=0.41)')
    plt.xlabel('Número de éxitos')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()

    # Funcion de densidad
    plt.figure(figsize=(8, 6))
    x = np.arange(binom.ppf(0.0001, n, p), binom.ppf(0.999, n, p))
    plt.plot(x, binom.pmf(x, n, p), '-r', ms=2)
    plt.title('Histograma de una variable aleatoria discreta con distribucion binomial')
    plt.xlabel('Valor de la variable')
    plt.ylabel('Ocurrencias')
    plt.ylabel("Densidad de ocurrencias")
    plt.show()

    test_chi_cuadrado_binomial(samples)

def graficar_empirica_discreta(size):
    # Suponé que tenés esta muestra empírica de valores discretos:
    samples = np.random.choice([0, 1, 2, 3, 4, 5], size=size, p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.1])

    # Histograma (gráfico de barras porque es discreta)
    plt.figure(figsize=(8, 6))
    sns.countplot(x=samples, color='skyblue')
    plt.title('Histograma de muestra empírica discreta')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()

    # Densidad discreta (probabilidad estimada)
    plt.figure(figsize=(8, 6))
    valores, cuentas = np.unique(samples, return_counts=True)
    probabilidades = cuentas / len(samples)
    plt.stem(valores, probabilidades, basefmt=" ")
    plt.title('Distribución de probabilidad empírica')
    plt.xlabel('Valor')
    plt.ylabel('Probabilidad estimada')

    plt.tight_layout()
    plt.show()

    test_chi_cuadrado_empirica_discreta(size, samples)

def test_chi_cuadrado_empirica_discreta(size, muestra):
    # Generar muestra empírica
    valores_posibles = [0, 1, 2, 3, 4, 5]
    probabilidades_teoricas = [0.1, 0.2, 0.3, 0.2, 0.1, 0.1]

    # Frecuencias observadas
    observadas = np.array([np.sum(muestra == val) for val in valores_posibles])

    # Frecuencias esperadas
    esperadas = np.array([p * size for p in probabilidades_teoricas])

    # Test de chi-cuadrado
    estadistico, p_valor = chisquare(f_obs=observadas, f_exp=esperadas)

    print("Estadístico chi-cuadrado:", estadistico)
    print("Valor p:", p_valor)

    alpha = 0.05
    if p_valor < alpha:
      print("No supera el test")
    else:
      print("Supera el test")

def graficar_exponencial(n, lamda):
    # Parámetros de la exponencial
    scale_param = 1 / lamda

    # Generar datos
    samples = np.random.exponential(scale=scale_param, size=n)

    # === GRÁFICOS ===

    plt.figure(figsize=(8, 6))

    # Histograma + curva teórica
    sns.histplot(samples, bins=50, kde=False, color='skyblue')
    plt.title('Histograma de muestra exponencial')
    plt.xlabel('Valor')
    plt.ylabel('Densidad')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    x = np.linspace(0, np.max(samples), 1000)
    plt.plot(x, expon.pdf(x, scale=scale_param), color='red')
    plt.title('Funcion de densidad')
    plt.xlabel('Valor')
    plt.ylabel('Densidad')
    plt.legend()
    plt.tight_layout()
    plt.show()

    test_chi_cuadrado_exponencial(n, lamda)

def test_chi_cuadrado_exponencial(n, lamda, bins=10):
    scale = 1 / lamda
    muestras = np.random.exponential(scale=scale, size=n)

    # Definir bordes de los intervalos (cuantiles teóricos)
    cuantiles = np.linspace(0, 1, bins + 1)
    bordes = expon.ppf(cuantiles, scale=scale)

    # Contar observaciones en cada intervalo
    observadas, _ = np.histogram(muestras, bins=bordes)

    # Frecuencias esperadas iguales en todos los intervalos
    esperadas = np.array([n / bins] * bins)

    # Test de chi-cuadrado
    estadistico, p_valor = chisquare(f_obs=observadas, f_exp=esperadas)

    print("Estadístico chi-cuadrado:", estadistico)
    print("Valor p:", p_valor)

    alpha = 0.05
    if p_valor < alpha:
      print("No supera el test")
    else:
      print("Supera el test")

def graficar_gamma(n):
    # Parámetros gamma
    shape_k = 2     # parámetro de forma (a)
    scale_theta = 2 # parámetro de escala

    # Generar datos
    samples = np.random.gamma(shape=shape_k, scale=scale_theta, size=n)

    # === GRÁFICOS ===

    plt.figure(figsize=(8, 6))

    # Histograma + función densidad teórica
    sns.histplot(samples, bins=50, kde=False, color='skyblue')

    plt.title('Histograma de muestra gamma')
    plt.xlabel('Valor')
    plt.ylabel('Densidad')
    plt.tight_layout()
    plt.show()

    x = np.linspace(0, np.max(samples), 1000)
    plt.plot(x, gamma.pdf(x, a=shape_k, scale=scale_theta), color='red')
    plt.title('Histograma de muestra gamma')
    plt.xlabel('Valor')
    plt.ylabel('Densidad')
    plt.tight_layout()
    plt.show()

    test_chi_cuadrado_gamma(n, shape_k, scale_theta)

def test_chi_cuadrado_gamma(n, k=2, theta=2, bins=10):
    muestras = np.random.gamma(shape=k, scale=theta, size=n)

    # Bordes de intervalos usando cuantiles teóricos
    cuantiles = np.linspace(0, 1, bins + 1)
    bordes = gamma.ppf(cuantiles, a=k, scale=theta)

    # Frecuencias observadas
    observadas, _ = np.histogram(muestras, bins=bordes)

    # Frecuencias esperadas (iguales)
    esperadas = np.array([n / bins] * bins)

    # Test chi-cuadrado
    estadistico, p_valor = chisquare(f_obs=observadas, f_exp=esperadas)

    print("Estadístico chi-cuadrado:", estadistico)
    print("Valor p:", p_valor)

    alpha = 0.05
    if p_valor < alpha:
      print("No supera el test")
    else:
      print("Supera el test")

def graficar_hipergeometrica(n, size):
    N = 500      # tamaño de la población
    K = 200      # número de éxitos en la población

    # Generar muestras
    samples = hypergeom.rvs(M=N, n=K, N=n, size=size)
    # Valores posibles que puede tomar la variable
    valores_posibles = np.arange(hypergeom.ppf(0.0001, N, K, n), hypergeom.ppf(0.9999, N, K, n) + 1).astype(int)
    observadas = np.array([np.sum(samples == k) for k in valores_posibles])

    prob_teoricas = hypergeom.pmf(valores_posibles, M=N, n=K, N=n)
    proporciones_esperadas = prob_teoricas  # ya son proporciones

    plt.figure(figsize=(8, 6))
    plt.bar(valores_posibles, observadas, width=0.4, color='skyblue', align='center')
    plt.title('Distribución hipergeométrica (proporciones)')
    plt.xlabel('Número de éxitos en la muestra')
    plt.ylabel('Frecuencia absoluta')
    plt.xticks(valores_posibles)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(valores_posibles, proporciones_esperadas, color='red', linewidth=2)
    plt.title('Distribución hipergeométrica (proporciones)')
    plt.xlabel('Número de éxitos en la muestra')
    plt.ylabel('Frecuencia relativa')
    plt.xticks(valores_posibles)
    plt.tight_layout()
    plt.show()

    test_chi_cuadrado_hipergeometrica(n, size, N, K, samples)

def test_chi_cuadrado_hipergeometrica(n, size, N, K, samples):
    valores_posibles = np.arange(max(0, n + K - N), min(K, n) + 1)

    observadas = np.array([np.sum(samples == k) for k in valores_posibles])
    prob_teoricas = hypergeom.pmf(valores_posibles, M=N, n=K, N=n)
    esperadas = prob_teoricas * size

    mask = esperadas >= 5
    observadas_validas = observadas[mask]
    esperadas_validas = esperadas[mask]

    # Ajustar suma de observadas para que coincida con esperadas
    factor_ajuste = esperadas_validas.sum() / observadas_validas.sum()
    observadas_validas_ajustadas = observadas_validas * factor_ajuste

    chi_stat, p_valor = chisquare(f_obs=observadas_validas_ajustadas, f_exp=esperadas_validas)

    print("Estadístico chi-cuadrado:", chi_stat)
    print("Valor p:", p_valor)
    alpha = 0.05
    if p_valor < alpha:
        print("No supera el test")
    else:
        print("Supera el test")

def graficar_normal(size):
    # Parámetros de la distribución normal
    mu = 10
    sigma = 2

    # Generar muestras normales
    samples = np.random.normal(loc=mu, scale=sigma, size=size)

    # === GRAFICAR HISTOGRAMA + CURVA DE DENSIDAD ===
    plt.figure(figsize=(8, 6))

    # Histograma con densidad normalizada
    _, bins, _ = plt.hist(samples, bins=30, alpha=0.6, color='blue', label='Histograma (normalizado)')
    plt.title('Distribución normal: histograma y densidad')
    plt.xlabel('Valor')
    plt.ylabel('Densidad')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))

    # Superponer la PDF teórica
    x = np.linspace(min(bins), max(bins), 1000)
    plt.plot(x, norm.pdf(x, loc=mu, scale=sigma), 'r-', lw=2)
    plt.title('Distribución normal: histograma y densidad')
    plt.xlabel('Valor')
    plt.ylabel('Densidad')
    plt.tight_layout()
    plt.show()

    test_chi_cuadrado_normal(size, mu, sigma)

def test_chi_cuadrado_normal(n, mu=10, sigma=2, bins=10):
    muestras = np.random.normal(loc=mu, scale=sigma, size=n)

    # Definir bordes por cuantiles teóricos
    cuantiles = np.linspace(0, 1, bins + 1)
    bordes = norm.ppf(cuantiles, loc=mu, scale=sigma)

    # Frecuencias observadas
    observadas, _ = np.histogram(muestras, bins=bordes)

    # Frecuencias esperadas iguales
    esperadas = np.array([n / bins] * bins)

    # Test de chi-cuadrado
    estadistico, p_valor = chisquare(f_obs=observadas, f_exp=esperadas)

    print("Estadístico chi-cuadrado:", estadistico)
    print("Valor p:", p_valor)

    alpha = 0.05
    if p_valor < alpha:
      print("No supera el test")
    else:
      print("Supera el test")

def graficar_pascal(size):
    # Parámetros de la distribución de Pascal (binomial negativa)
    r = 5        # número de éxitos deseados
    p = 0.4      # probabilidad de éxito

    # Generar muestras
    samples = nbinom.rvs(r, p, size=size)

    # Obtener valores posibles en el rango observado
    valores_posibles = np.arange(0, samples.max() + 1)

    # Frecuencia observada
    observadas = np.array([np.sum(samples == k) for k in valores_posibles])

    # PMF teórica y frecuencias esperadas
    pmf = nbinom.pmf(valores_posibles, r, p)
    esperadas = pmf * size

    # === Gráfico ===
    plt.figure(figsize=(8, 6))
    plt.bar(valores_posibles, observadas, color='skyblue', width=0.5)
    plt.title('Distribución de Pascal (Binomial Negativa)')
    plt.xlabel('Cantidad de fracasos antes de lograr {} éxitos'.format(r))
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(valores_posibles, pmf, 'r-', linewidth=2)
    plt.title('Distribución de Pascal (Binomial Negativa)')
    plt.xlabel('Cantidad de fracasos antes de lograr {} éxitos'.format(r))
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()

def graficar_poisson(size):
    mu = 3
    samples = poisson.rvs(mu, size=size)

    # Punto de corte para agrupar la cola
    cutoff = max(10, samples.max())
    valores_posibles = np.arange(0, cutoff)

    # Observadas
    observadas = np.array([np.sum(samples == k) for k in valores_posibles])
    observadas = np.append(observadas, np.sum(samples >= cutoff))

    # Esperadas
    pmf = poisson.pmf(valores_posibles, mu)
    ultima_prob = 1 - poisson.cdf(cutoff - 1, mu)
    pmf = np.append(pmf, ultima_prob)
    esperadas = pmf * size

    # Crear máscara antes del ajuste
    mask = esperadas >= 5

    # Ajustar esperadas solo para los valores enmascarados
    obs_sum = observadas[mask].sum()
    exp_sum = esperadas[mask].sum()

    # Reescalar esperadas[mask] y sobreescribir en esperadas
    escala = obs_sum / exp_sum
    esperadas[mask] *= escala

    # Corregir última celda enmascarada para forzar igualdad exacta
    diferencia = observadas[mask].sum() - esperadas[mask].sum()
    if abs(diferencia) > 1e-10:
        idx_last = np.where(mask)[0][-1]
        esperadas[idx_last] += diferencia

    # Test chi-cuadrado
    estadistico, p_valor = chisquare(f_obs=observadas[mask], f_exp=esperadas[mask])

    print("Estadístico chi-cuadrado:", estadistico)
    print("Valor p:", p_valor)
    alpha = 0.05
    if p_valor < alpha:
        print("No supera el test")
    else:
        print("Supera el test")

    # Gráficos
    etiquetas = list(map(str, valores_posibles)) + [f"{cutoff}+"]
    x = np.arange(len(etiquetas))

    plt.figure(figsize=(8, 6))
    plt.bar(x, observadas, color='skyblue')
    plt.xticks(x, etiquetas, rotation=45)
    plt.title(f'Distribución de Poisson (μ = {mu})')
    plt.xlabel('Valor (última clase = resto)')
    plt.ylabel('Frecuencia observada')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(x, pmf, 'r-', linewidth=2)
    plt.xticks(x, etiquetas, rotation=45)
    plt.title(f'Distribución de Poisson (μ = {mu})')
    plt.xlabel('Valor (última clase = resto)')
    plt.ylabel('Probabilidad teórica')
    plt.tight_layout()
    plt.show()

def test_chi_cuadrado_poisson(n, mu):
    muestras = poisson.rvs(mu, size=n)

    valores = np.arange(0, muestras.max() + 1)
    observadas = np.array([np.sum(muestras == k) for k in valores])

    probabilidades = poisson.pmf(valores, mu)
    esperadas = probabilidades * n

    # Ajustar esperadas para que sumen lo mismo que observadas
    esperadas *= observadas.sum() / esperadas.sum()

    # Filtrar clases con esperadas < 5
    mask = esperadas >= 5
    chi2_stat, p_value = chisquare(observadas[mask], f_exp=esperadas[mask])
    print("Chi² =", chi2_stat)
    print("p-value =", p_value)
    print("✅ H0 NO se rechaza" if p_value >= 0.05 else "❌ Se rechaza H0")

def graficar_uniforme(size):
    # Parámetros
    a = 0
    b = 10  # genera valores de 0 a 9

    # Generar muestras
    samples = randint.rvs(a, b, size=size)

    valores_posibles = np.arange(a, b)
    observadas = np.array([np.sum(samples == k) for k in valores_posibles])
    pmf = randint.pmf(valores_posibles, a, b)
    esperadas = pmf * size

    # Gráfico
    plt.figure(figsize=(8, 6))
    plt.bar(valores_posibles, observadas, color='lightgreen')
    plt.title('Distribución Uniforme Discreta [{} - {})'.format(a, b))
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()


    plt.plot(valores_posibles, pmf, 'r-', linewidth=2)
    plt.title('Distribución Uniforme Discreta [{} - {})'.format(a, b))
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()

    test_chi_cuadrado_uniforme(size, a, b)


def test_chi_cuadrado_uniforme(size, a=0, b=10):
    muestras = randint.rvs(a, b, size=size)

    valores = np.arange(a, b)
    observadas = np.array([np.sum(muestras == k) for k in valores])

    pmf = randint.pmf(valores, a, b)
    esperadas = pmf * size

    # Ajustar esperadas para que sumen lo mismo que observadas
    esperadas *= observadas.sum() / esperadas.sum()

    estadistico, p_valor = chisquare(f_obs=observadas, f_exp=esperadas)

    print("Estadístico chi-cuadrado:", estadistico)
    print("Valor p:", p_valor)

    alpha = 0.05
    if p_valor < alpha:
      print("No supera el test")
    else:
      print("Supera el test")

n = 100
p = 0.41
size = 1000
lamba = 1

graficar_binomial(n, p, size)
graficar_empirica_discreta(size)
graficar_exponencial(size, lamba)
graficar_gamma(size)
n = 100
size = 10000
graficar_hipergeometrica(n, size)
graficar_normal(size)
graficar_pascal(size)
size = 1000
graficar_poisson(size)