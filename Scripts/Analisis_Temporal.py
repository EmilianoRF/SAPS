#%% Librerías
# -*- coding: utf-8 -*-
"""

Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Script ejemplificando el uso de la función provista para levanta los registros
almacenados en .csv, el maneho de las señales y su graficación. 

Autor: Albano Peñalva
Fecha: Marzo 2022
"""

# Librerías
import process_data
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
#%% Lectura del dataset
plt.close('all')
FS = 500 # Frecuencia de muestre: 500Hz
T = 3    # Tiempo total de cada registro: 2 segundos

folder = 'dataset_figuras' # Carpeta donde se almacenan los .csv

x, y, z, classmap = process_data.process_data(FS, T, folder)

#%% Graficación

ts = 1 / FS                     # tiempo de muestreo
N = FS*T                        # número de muestras en cada regsitro
t = np.linspace(0, N * ts, N)   # vector de tiempos

# Se crea un arreglo de gráficas, con tres columnas de gráficas 
# (correspondientes a cada eje) y tantos renglones como gesto distintos.
fig, axes = plt.subplots(len(classmap), 3, figsize=(20, 20))
fig.subplots_adjust(hspace=0.5)

# Se recorren y grafican todos los registros
trial_num = 0
for gesture_name in classmap:                           # Se recorre cada gesto
    for capture in range(int(len(x))):                  # Se recorre cada renglón de las matrices
        if (x[capture, N] == gesture_name):             # Si en el último elemento se detecta la etiqueta correspondiente
            # Se grafica la señal en los tres ejes
            axes[gesture_name][0].plot(t, x[capture, 0:N], label="Trial {}".format(trial_num))
            axes[gesture_name][1].plot(t, y[capture, 0:N], label="Trial {}".format(trial_num))
            axes[gesture_name][2].plot(t, z[capture, 0:N], label="Trial {}".format(trial_num))
            trial_num = trial_num + 1
    trial_num = 0
# Se le da formato a los ejes de cada gráfica
    axes[gesture_name][0].set_title(classmap[gesture_name] + " (Aceleración X)")
    axes[gesture_name][0].grid(linestyle='dashed')
    axes[gesture_name][0].legend(fontsize=6, loc='upper right');
    axes[gesture_name][0].set_xlabel('Tiempo [s]', fontsize=10)
    axes[gesture_name][0].set_ylabel('Aceleración [G]', fontsize=10)
    
    axes[gesture_name][1].set_title(classmap[gesture_name] + " (Aceleración Y)")
    axes[gesture_name][1].grid(linestyle='dashed')
    axes[gesture_name][1].legend(fontsize=6, loc='upper right');
    axes[gesture_name][1].set_xlabel('Tiempo [s]', fontsize=10)
    axes[gesture_name][1].set_ylabel('Aceleración [G]', fontsize=10)
    
    axes[gesture_name][2].set_title(classmap[gesture_name] + " (Aceleración Z)")
    axes[gesture_name][2].grid(linestyle='dashed')
    axes[gesture_name][2].legend(fontsize=6, loc='upper right');
    axes[gesture_name][2].set_xlabel('Tiempo [s]', fontsize=10)
    axes[gesture_name][2].set_ylabel('Aceleración [G]', fontsize=10)

plt.tight_layout()
plt.show()

#%% Análisis frecuencial

fig.subplots_adjust(hspace=0.5)

freq = fft.fftfreq(N, d=1/FS)
# El espectro es simétrico, nos quedamos solo con el semieje positivo
f  = freq[np.where(freq >= 0)]   
# Se recorren y grafican todos los registros
trial_num = 0
for gesture_name in classmap:
    fig, axes = plt.subplots(3,1, figsize=(20, 20))     # Se recorre cada gesto
    for capture in range(int(len(x))):                  # Se recorre cada renglón de las matrices
        if (x[capture, N] == gesture_name):             # Si en el último elemento se detecta la etiqueta correspondiente
            # Se grafica la señal en los tres ejes            

            # Se calcula la transformada rápida de Fourier
            fft_x  = fft.fft(x[capture, 0:N])
            fft_y  = fft.fft(y[capture, 0:N])
            fft_z  = fft.fft(z[capture, 0:N])   

            # Se calcula la magnitud del espectro 
            ''' Respetando la relación de Parseval. Al haberse descartado la mitad del espectro, para conservar la energía 
                original de la señal, se debe multiplicar la mitad restante por dos (excepto en 0 y fm/2)
            '''
            mod_fft_x = np.abs(fft_x) / N
            mod_fft_x[1:len(mod_fft_x-1)] = 2 * mod_fft_x[1:len(mod_fft_x-1)]
            
            mod_fft_y = np.abs(fft_y) / N
            mod_fft_y[1:len(mod_fft_y-1)] = 2 * mod_fft_y[1:len(mod_fft_y-1)]
            
            mod_fft_z = np.abs(fft_z) / N
            mod_fft_z[1:len(mod_fft_z-1)] = 2 * mod_fft_z[1:len(mod_fft_z-1)]
            
            axes[0].plot(f,mod_fft_x[0:round(N/2)], label="Trial {}".format(trial_num))
            axes[1].plot(f,mod_fft_y[0:round(N/2)], label="Trial {}".format(trial_num))
            axes[2].plot(f,mod_fft_z[0:round(N/2)], label="Trial {}".format(trial_num))
            trial_num = trial_num + 1

        # Se le da formato a los ejes de cada gráfica
            axes[0].set_title(classmap[gesture_name] + " |FFT{Aceleración X}|")
            axes[0].grid(linestyle='dashed')
            axes[0].legend(fontsize=6, loc='upper right');
            axes[0].set_xlabel('Frecuencia [Hz]', fontsize=10)
            axes[0].set_ylabel('Magnitud', fontsize=10)
            axes[0].set_xlim(0,51)
         
            axes[1].set_title(classmap[gesture_name] + " |FFT{Aceleración Y}|")
            axes[1].grid(linestyle='dashed')
            axes[1].legend(fontsize=6, loc='upper right');
            axes[1].set_xlabel('Frecuencia [Hz]', fontsize=10)
            axes[1].set_ylabel('Magnitud', fontsize=10)
            axes[1].set_xlim(0,51)

            axes[2].set_title(classmap[gesture_name] + " |FFT{Aceleración Z}|")
            axes[2].grid(linestyle='dashed')
            axes[2].legend(fontsize=6, loc='upper right');
            axes[2].set_xlabel('Frecuencia [Hz]', fontsize=10)
            axes[2].set_ylabel('Magnitud', fontsize=10)
            axes[2].set_xlim(0,51)

    trial_num = 0
    plt.tight_layout()
    plt.show()


