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
    plt.title('Histograma para la distribución binomial')
    plt.xlabel('Número de éxitos')
    plt.ylabel('Frecuencia absoulta')
    plt.tight_layout()
    plt.savefig(f'./images/TP2.2/binomial histograma.png')
    plt.show()

    # Funcion de densidad
    plt.figure(figsize=(8, 6))
    x = np.arange(binom.ppf(0.0001, n, p), binom.ppf(0.999, n, p))
    plt.plot(x, binom.pmf(x, n, p), '-r', ms=2)
    plt.title('Función de densidad para la distribución binomial')#'Histograma de una variable aleatoria discreta con distribucion binomial
    plt.xlabel('Valor de la variable')
    plt.ylabel("Frecuendia relativa")
    plt.savefig(f'./images/TP2.2/binomial densidad.png')
    plt.show()

    test_chi_cuadrado_binomial(samples)

def graficar_empirica_discreta(size):
    # Suponé que tenés esta muestra empírica de valores discretos:
    samples = np.random.choice([0, 1, 2, 3, 4, 5], size=size, p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.1])

    # Histograma (gráfico de barras porque es discreta)
    plt.figure(figsize=(8, 6))
    sns.countplot(x=samples, color='skyblue')
    plt.title('Histograma para la distribución empírica discreta')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia absoluta')
    plt.savefig(f'./images/TP2.2/empirica discreta histograma.png')
    plt.tight_layout()
    plt.show()

    # Densidad discreta (probabilidad estimada)
    plt.figure(figsize=(8, 6))
    valores, cuentas = np.unique(samples, return_counts=True)
    probabilidades = cuentas / len(samples)
    plt.stem(valores, probabilidades, basefmt=" ")
    plt.title('Densidad para la distribución empírica discreta')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia relativa')
    plt.savefig(f'./images/TP2.2/empirica discreta densidad.png')
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
    plt.title('Histograma para la distribución exponencial')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia absoluta')
    plt.tight_layout()
    plt.savefig(f'./images/TP2.2/exponencial histograma.png')
    plt.show()

    plt.figure(figsize=(8, 6))
    x = np.linspace(0, np.max(samples), 1000)
    plt.plot(x, expon.pdf(x, scale=scale_param), color='red')
    plt.title('Función de densidad para la distribución exponencial')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia relativa')
    plt.tight_layout()
    plt.savefig(f'./images/TP2.2/exponencial densidad.png')
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

    plt.title('Histograma para la distribución gamma')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia absoluta')
    plt.savefig(f'./images/TP2.2/gamma histograma.png')
    plt.tight_layout()
    plt.show()

    x = np.linspace(0, np.max(samples), 1000)
    plt.plot(x, gamma.pdf(x, a=shape_k, scale=scale_theta), color='red')
    plt.title('Función de densidad para la distribución gamma')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia relativa')
    plt.savefig(f'./images/TP2.2/gamma densidad.png')
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

    # Crear histograma real a partir de los datos individuales
    plt.figure(figsize=(8, 6))
    plt.hist(samples, bins=np.arange(samples.min(), samples.max()+2)-0.5,
             color='skyblue', edgecolor='black')
    plt.title('Histograma para la distribución Hipergeométrica')
    plt.xlabel('Número de éxitos en la muestra')
    plt.ylabel('Frecuencia absoluta')
    plt.xticks(np.arange(samples.min(), samples.max()+1))
    plt.tight_layout()
    plt.savefig('./images/TP2.2/hipergeometrica_histograma.png')
    plt.show()

    # Calcular valores teóricos para la distribución
    valores_posibles = np.arange(hypergeom.ppf(0.0001, N, K, n), 
                                 hypergeom.ppf(0.9999, N, K, n) + 1).astype(int)
    proporciones_esperadas = hypergeom.pmf(valores_posibles, M=N, n=K, N=n)

    # Gráfico de la función de masa de probabilidad (teórica)
    plt.figure(figsize=(8, 6))
    plt.plot(valores_posibles, proporciones_esperadas, color='red', linewidth=2)
    plt.title('Función de densidad para la distribución Hipergeométrica')
    plt.xlabel('Número de éxitos en la muestra')
    plt.ylabel('Frecuencia relativa')
    plt.xticks(valores_posibles)
    plt.tight_layout()
    plt.savefig('./images/TP2.2/hipergeometrica_densidad.png')
    plt.show()

    # Llamar a función de prueba estadística (si existe)
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
    _, bins, _ = plt.hist(samples, bins=30, alpha=0.6, color='blue', label='Histograma (normalizado)', edgecolor='black')
    plt.title('Histograma para la distribución normal')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia absoluta')
    plt.savefig(f'./images/TP2.2/normal histograma.png')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))

    # Superponer la PDF teórica
    x = np.linspace(min(bins), max(bins), 1000)
    plt.plot(x, norm.pdf(x, loc=mu, scale=sigma), 'r-', lw=2)
    plt.title('Función de densidad de la distribución normal')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia relativa')
    plt.savefig(f'./images/TP2.2/normal densidad.png')
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

    # === Histograma real ===
    plt.figure(figsize=(8, 6))
    plt.hist(samples, bins=np.arange(samples.min(), samples.max() + 2) - 0.5,
             color='skyblue', edgecolor='black')
    plt.title('Histograma para la distribución Pascal (Binomial Negativa)')
    plt.xlabel('Cantidad de fracasos antes de lograr {} éxitos'.format(r))
    plt.ylabel('Frecuencia absoluta')
    plt.xticks(np.arange(samples.min(), samples.max() + 1))
    plt.tight_layout()
    plt.savefig('./images/TP2.2/pascal_histograma.png')
    plt.show()

    # === Gráfico de la distribución teórica (PMF) ===
    valores_posibles = np.arange(0, samples.max() + 1)
    pmf = nbinom.pmf(valores_posibles, r, p)

    plt.figure(figsize=(8, 6))
    plt.plot(valores_posibles, pmf, 'r-', linewidth=2)
    plt.title('Función de densidad de la distribución Pascal (Binomial Negativa)')
    plt.xlabel('Cantidad de fracasos antes de lograr {} éxitos'.format(r))
    plt.ylabel('Frecuencia relativa')
    plt.xticks(valores_posibles)
    plt.tight_layout()
    plt.savefig('./images/TP2.2/pascal_densidad.png')
    plt.show()


def graficar_poisson(size):
    mu = 3
    samples = poisson.rvs(mu, size=size)

    # === Histograma real ===
    plt.figure(figsize=(8, 6))
    plt.hist(samples, bins=np.arange(samples.min(), samples.max() + 2) - 0.5,
             color='skyblue', edgecolor='black')
    plt.title(f'Histograma para la distribución de Poisson (μ = {mu})')
    plt.xlabel('Valores')
    plt.ylabel('Frecuencia absoluta')
    plt.xticks(np.arange(samples.min(), samples.max() + 1))
    plt.tight_layout()
    plt.savefig('./images/TP2.2/poisson_histograma.png')
    plt.show()

    # === Preparación para test chi-cuadrado ===
    cutoff = max(10, samples.max())
    valores_posibles = np.arange(0, cutoff)
    observadas = np.array([np.sum(samples == k) for k in valores_posibles])
    observadas = np.append(observadas, np.sum(samples >= cutoff))

    pmf = poisson.pmf(valores_posibles, mu)
    ultima_prob = 1 - poisson.cdf(cutoff - 1, mu)
    pmf = np.append(pmf, ultima_prob)
    esperadas = pmf * size

    mask = esperadas >= 5
    obs_sum = observadas[mask].sum()
    exp_sum = esperadas[mask].sum()
    escala = obs_sum / exp_sum
    esperadas[mask] *= escala
    diferencia = observadas[mask].sum() - esperadas[mask].sum()
    if abs(diferencia) > 1e-10:
        idx_last = np.where(mask)[0][-1]
        esperadas[idx_last] += diferencia

    estadistico, p_valor = chisquare(f_obs=observadas[mask], f_exp=esperadas[mask])
    print("Estadístico chi-cuadrado:", estadistico)
    print("Valor p:", p_valor)
    print("✅ Supera el test" if p_valor >= 0.05 else "❌ No supera el test")

    # === Gráfico de barras (frecuencias observadas agrupadas) ===
    etiquetas = list(map(str, valores_posibles)) + [f"{cutoff}+"]
    x = np.arange(len(etiquetas))

    # === Densidad teórica ===
    plt.figure(figsize=(8, 6))
    plt.plot(x, pmf, 'r-', linewidth=2)
    plt.xticks(x, etiquetas, rotation=45)
    plt.title(f'Función de densdidad de la distribución de Poisson (μ = {mu})')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia relativa')
    plt.tight_layout()
    plt.savefig('./images/TP2.2/poisson_densidad.png')
    plt.show()



def graficar_uniforme(size):
    # Parámetros
    a = 0
    b = 10  # genera valores de 0 a 9

    # Generar muestras
    samples = randint.rvs(a, b, size=size)

    # === Histograma absoluto ===
    plt.figure(figsize=(8, 6))
    plt.hist(samples, bins=np.arange(a, b + 1) - 0.5, 
             color='lightgreen', edgecolor='black')
    plt.title('Histograma para la distribución Uniforme Discreta [{} - {}]'.format(a, b))
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia absoluta')
    plt.xticks(np.arange(a, b))
    plt.tight_layout()
    plt.savefig(f'./images/TP2.2/uniforme_histograma.png')
    plt.show()

    # === Gráfico de frecuencias relativas ===
    valores_posibles = np.arange(a, b)
    hist, _ = np.histogram(samples, bins=np.arange(a, b + 1))
    frecuencia_relativa = hist / size

    plt.figure(figsize=(8, 6))
    plt.plot(valores_posibles, frecuencia_relativa, 'r-', linewidth=2)
    plt.title('Función de densdidad de la distribución Uniforme discreta [{} - {})'.format(a, b))
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia relativa')
    plt.xticks(valores_posibles)
    plt.ylim(0, max(frecuencia_relativa) * 1.1)
    plt.tight_layout()
    plt.savefig(f'./images/TP2.2/uniforme_frecuencia_relativa.png')
    plt.show()

    # Test chi-cuadrado
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
size = 10000
lamba = 1

graficar_binomial(n, p, size)
graficar_empirica_discreta(size)
graficar_exponencial(size, lamba)
graficar_gamma(size)
graficar_hipergeometrica(n, size)
graficar_normal(size)
graficar_pascal(size)
graficar_poisson(size)
graficar_uniforme(size)