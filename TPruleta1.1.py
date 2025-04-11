import random
import matplotlib.pyplot as plt
import numpy as np

def create_box_plot_graph(number_samples):
    min_val = np.min(number_samples)
    q1 = np.percentile(number_samples, 25)
    median = np.median(number_samples)
    q3 = np.percentile(number_samples, 75)
    max_val = np.max(number_samples)

    plt.boxplot(number_samples, patch_artist=True,
            boxprops=dict(facecolor='skyblue', color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            medianprops=dict(color='red', linewidth=1.5))

    plt.annotate('Valor Mínimo: ' + str(min_val), xy=(1, min_val), xytext=(1.25, min_val),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.annotate('Q1: ' + str(round(q1, 1)), xy=(1, q1), xytext=(1.25, q1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.annotate('Mediana: ' + str(round(median, 1)), xy=(1, median), xytext=(1.25, median),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.annotate('Q3: ' + str(round(q3, 1)), xy=(1, q3), xytext=(1.25, q3),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.annotate('Valor Máximo: ' + str(max_val), xy=(1, max_val), xytext=(1.25, max_val),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.xlim(0.8, 1.5)

    plt.title('Diagrama de caja de las tiradas de la ruleta')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def create_relative_frequency_graph(trials, number_samples):
    number_dict = {}
    for num in number_samples:
        if num in number_dict:
            number_dict[num] += 1
        else:
            number_dict[num] = 1

    plt.plot([(value) for value in number_dict], [number_dict[value] for value in number_dict])

    avg_frequency = sum(number_dict.values()) / len(number_dict)

    plt.axhline(y=avg_frequency, color='r', linestyle='--', label=f'Promedio: {avg_frequency:.2f}')

    key_max_value, max_value = max(number_dict.items(), key=lambda item: item[1])
    key_min_value, min_value = min(number_dict.items(), key=lambda item: item[1])
    plt.annotate('Frecuencia máxima:' + str(max_value), xy= (key_max_value, max_value),
                  xytext = (key_max_value - 10 if key_max_value > 15 else key_max_value + 10, max_value),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    plt.annotate('Frecuencia mínima:' + str(min_value), xy= (key_min_value, min_value),
                  xytext = (key_min_value - 10 if key_min_value > 15 else key_min_value + 10, min_value), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    plt.title("Frecuencia relativa de las tiradas")
    plt.xlabel(str(trials) + "(número de tiradas)")
    plt.ylabel("fr(frecuencia relativa)")
    plt.xlim(0.0, 36.0)
    plt.legend()
    plt.show()

trials = 1000
colors = ["red", "black"]
numbers = range(0, 37)

color_samples = [(random.choice(colors)) for _ in range(trials)]
number_samples = [random.choice(numbers) for _ in range(trials)]
number_samples.sort()

create_box_plot_graph(number_samples)
create_relative_frequency_graph(trials, number_samples)