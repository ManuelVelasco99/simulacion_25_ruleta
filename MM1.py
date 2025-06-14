import math
import random
import pandas as pd
import matplotlib.pyplot as plt

Q_LIMIT = 2#int(input('Ingrese capacidad maxima de la cola: '))
BUSY = 1    # Indica si el servidor esta ocupado
IDLE = 0    # Indica si el servidor esta inactivo

time_arrival = [0.0] * (Q_LIMIT + 1)
time_next_event = [0.0] * 3

num_in_queue_counts = [0] * (Q_LIMIT + 1)


def initialize():
    global sim_time, server_status, num_in_q, time_last_event, num_custs_delayed, total_of_delays, area_num_in_q, area_server_status
    global time_next_event
    global num_in_queue_counts, num_in_system, total_time_in_system
    global area_num_in_system, num_rejections

    # Inicializa el reloj de simulación
    sim_time = 0.0

    # Inicializa las variables de estado
    server_status = IDLE
    num_in_q = 0
    time_last_event = 0.0

    # Inicializa los contadores estadísticos
    num_custs_delayed = 0
    total_of_delays = 0.0
    area_num_in_q = 0.0
    area_server_status = 0.0

    # Inicializa la lista de eventos. Como no hay clientes presentes, el fin del servicio se elimina de
    # la consideración.
    time_next_event[1] = sim_time + expon(mean_interarrival)
    time_next_event[2] = 1.0e+30

    # Inicializa las nuevas variables extras
    num_in_queue_counts = [0] * (Q_LIMIT + 1)       # Número de clientes en cola
    num_in_system = 0                               # Número de clientes en el sistema
    area_num_in_system = 0.0                        # Área debajo de la función de clientes en el sistema
    total_time_in_system = 0.0                      # Tiempo total en el sistema
    num_rejections = 0
    global size_queue
    size_queue= pd.DataFrame({'time':[], 'size':[]})


def timing():
    global next_event_type, sim_time

    min_time_next_event = 1.0e+29
    next_event_type = 0

    # Determina el siguiente tipo de evento a ocurrir
    for i in range(1, num_events + 1):
        if time_next_event[i] < min_time_next_event:
            min_time_next_event = time_next_event[i]
            next_event_type = i

    # Comprueba si la lista de eventos está vacía. Si lo está, termina la simulación
    if next_event_type == 0:
        print("\nEvent list empty at time", sim_time)
        exit(1)

    # Si no está vacía, se avanza el reloj de simulación
    sim_time = min_time_next_event


def arrive():
    global num_in_q, server_status, total_of_delays, num_custs_delayed, time_next_event
    global num_in_system, num_rejections

    # Programa la siguiente llegada
    time_next_event[1] = sim_time + expon(mean_interarrival)
    num_in_system += 1
    # Comprueba si el servidor está ocupado
    if server_status == BUSY:

        # Si está ocupado, aumenta el número de clientes en cola
        num_in_q += 1


        # Comprueba si la cola está desbordada. Si lo está, finaliza la simulación.
        if num_in_q > Q_LIMIT:
            # print("\nOverflow of the array time_arrival at time", sim_time)
            # exit(2)

            # Suponiendo que en vez de finalizar la simulación, simplemente se
            # rechaza al cliente y se continúa con la misma cantidad que ya estaba:
            num_in_q -= 1
            num_rejections += 1

        # Hay espacio en la cola ya que no está desbordada. Luego, se almacena la hora de
        # llegada del cliente que llega al final de time_arrival
        time_arrival[num_in_q] = sim_time

    else:

        # El servidor está inactivo, el cliente que llega tiene un retraso de cero.
        delay = 0.0
        total_of_delays += delay

        # Incrementa el número de clientes retrasados y hace que el servidor esté ocupado.
        num_custs_delayed += 1
        server_status = BUSY

        # Programa una salida (fin de servicio)
        time_next_event[2] = sim_time + expon(mean_service)


def depart():
    global num_in_q, server_status, total_of_delays, num_custs_delayed, time_next_event
    global num_in_system, total_time_in_system

    # Comprueba si la cola está vacía
    if num_in_q == 0:

        # La cola está vacía: el servidor está inactivo y se elimina el evento de fin de servicio
        server_status = IDLE
        time_next_event[2] = 1.0e+30

    else:

        # La cola no está vacía: se disminuye el número de clientes en la cola
        num_in_q -= 1

        # Disminuye el número de clientes en el sistema
        if num_in_system > 0:
            num_in_system -= 1
        
        # Se calcula la demora del cliente que está iniciando el servicio y se
        # actualiza el acumulador de demora total
        delay = sim_time - time_arrival[1]
        total_of_delays += delay

        # Aumenta el tiempo total de clientes en el sistema
        total_time_in_system += delay
        
        # Incrementa el número de clientes retrasados y se programa la salida
        num_custs_delayed += 1
        time_next_event[2] = sim_time + expon(mean_service)
        
        # Mueve todos los clientes en cola (si los hay) un lugar hacia arriba
        for i in range(1, num_in_q + 1):
            time_arrival[i] = time_arrival[i + 1]


def update_time_avg_stats():
    global time_last_event, area_num_in_q, area_server_status, num_in_q, server_status, sim_time
    global area_num_in_system

    # Calcula el tiempo desde el último evento y actualiza el tiempo del último evento
    time_since_last_event = sim_time - time_last_event
    time_last_event = sim_time
    
    # Actualiza el área bajo la función del número en cola
    area_num_in_q += num_in_q * time_since_last_event
    
    global size_queue
    size_queue=pd.concat([size_queue, pd.DataFrame({'time':sim_time, 'size':num_in_q}, index=[0])], ignore_index=True)

    # Actualiza el área bajo la función de indicador de servidor ocupado
    area_server_status += server_status * time_since_last_event

    # Actualiza el área bajo la función de números de clientes en el sistema y
    # actualiza el número de clientes en cola
    num_in_queue_counts[num_in_q] += time_since_last_event
    area_num_in_system += num_in_system * time_since_last_event


def expon(mean):
    # Devuelve una variable aleatoria exponencial con media "mean"
    return -mean * math.log(random.random())


def simulation():
    # Inicia la simulación
    initialize()

    # Ejecuta la simulación hasta que un evento de fin (de tipo 3) ocurra. 
    while num_custs_delayed < num_delays_required:
        timing()    # Determina el siguiente evento
        update_time_avg_stats() # Actualiza los acumuladores estadísticos del promedio de tiempo

        if next_event_type == 1:
            arrive()
        elif next_event_type == 2:
            depart()


def main():
    global num_delays_required, mean_interarrival, mean_service, num_events
    # Ingreso de la media de entre llegadas, la media del servicio y el número de retrasos necesarios

    mean_interarrival = 2#float(input("Ingrese el tiempo promedio entre arribos: "))
    mean_service = 0.5#float(input("Ingrese el tiempo promedio de servicio: "))
    num_delays_required = 1000#int(input("Ingrese la cantidad de clientes: "))
    num_events = 2

    sum_Average_delay_in_queue = 0
    sum_Average_number_in_queue = 0
    sum_Server_utilization = 0
    sum_Average_number_of_customers_in_the_system = 0
    sum_Average_time_in_the_system = 0
    sum_Rejection_probabilty = 0
    for i in range(30):
        print('Corrida ',i+1,'/10')
        simulation()
        sum_Average_delay_in_queue += round(total_of_delays / num_custs_delayed, 3)
        sum_Average_number_in_queue += round(area_num_in_q / sim_time, 3)
        sum_Server_utilization += round(area_server_status / sim_time, 3)
        sum_Average_number_of_customers_in_the_system += round((area_num_in_q+ area_server_status) / sim_time, 3)
        sum_Average_time_in_the_system += round((total_of_delays / num_custs_delayed) + (1 / mean_service) , 3)
        sum_Rejection_probabilty += round(num_rejections * 100 / num_delays_required, 2)

    print('Cantidad promedio en sistema: ', sum_Average_number_of_customers_in_the_system/10)
    print('Cantidad promedio en cola: ', sum_Average_number_in_queue/10)
    print('Tiempo promedio en sistema: ',sum_Average_time_in_the_system/10)
    print('Espera promedio en cola: ',sum_Average_delay_in_queue/10)
    print('Utilizacion del servidor: ', sum_Server_utilization/10)
    print('Probabilidad de rechazo por cola llena: ',sum_Rejection_probabilty/10)
    print('Tiempo de simulación: ', sim_time)
    print('Número de clientes atendidos: ', num_custs_delayed)
    print('Número de rechazos: ', num_rejections)
    print('Número de clientes en el sistema: ', num_in_system)

    if input('Mostrar graficas? [y/n] ') == 'y':
        size_queue.plot(x='time', y='size', kind='line', title="tamaño de la cola en el tiempo")
        fig, ax = plt.subplots()
        arr_sum = sum(num_in_queue_counts)
        frecuency_queue_size = [item / arr_sum for item in num_in_queue_counts]
        ax.bar(range(len(frecuency_queue_size)), frecuency_queue_size, align='center', alpha=0.7)
        ax.set_xlabel('tamaño de cola')
        ax.set_ylabel('probabliidad')
        ax.set_title('Distribucion de probabilidades del tamaño de la cola')
        plt.show()

main()