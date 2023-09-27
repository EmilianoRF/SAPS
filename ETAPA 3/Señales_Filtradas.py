# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 20:22:05 2023

@author: Emiliano Riffel
"""

# Librerías
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import funciones_fft
from time import time
import sympy as sy
import process_data
import filter_parameters

plt.close('all') # cerrar gráficas anteriores

FS = 500 # Frecuencia de muestre: 500Hz
T = 3    # Tiempo total de cada registro: 2 segundos
folder = 'hechizos' # Carpeta donde se almacenan los .csv

x, y, z, classmap = process_data.process_data(FS, T, folder)

FS_resample = 40
decimate_factor = int(FS/FS_resample) 

ts = 1 / FS                     # tiempo de muestreo
N = FS*T                        # número de muestras en cada regsitro
t = np.linspace(0, N * ts, N)   # vector de tiempos
t_resampled = signal.decimate(t, decimate_factor)

# Carga de los Filtros 
# Se cargan los archivo generado mediante pyFDA
filtro_fir = np.load('fir_equiriple.npz', allow_pickle=True)
filtro_iir = np.load('iir_buttherworth.npz', allow_pickle=True) 

# Se extraen los coeficientes de numerador y denominador
Num_fir, Den_fir = filtro_fir['ba']     
Num_iir, Den_iir = filtro_iir['ba'] 
# Se expresan las funciones de transferencias (H(z))
Z = sy.Symbol('Z') # Se crea una variable simbólica z
Hz = sy.Symbol('H(Z)')
Numz_iir = 0
Denz_iir = 0
for i in range(len(Num_iir)): # Se arma el polinomio del numerador
    Numz_iir += Num_iir[i] * np.power(Z, -i)
for i in range(len(Den_iir)): # Se arma el polinomio del denominador
    Denz_iir += Den_iir[i] * np.power(Z, -i)
    

# Se evalúa la atenuación en las frecuncias de interés (a modo de ejemplo se 
# utilizan las frecuencia de la interferencia y del primer armónico)
_, h1_fir = signal.freqz(Num_fir, Den_fir, worN=[1, 50], fs=FS_resample)
_, h1_iir = signal.freqz(Num_iir, Den_iir, worN=[1, 50], fs=FS_resample)


# Se extraen polos y ceros de los filtros
zeros_fir, polos_fir, k_fir =   filtro_fir['zpk']  
zeros_iir, polos_iir, k_iir =   filtro_iir['zpk']

# Se recorren y grafican todos los registros
trial_num = 0
Xprom = []
Xprom_iir = []
Xprom_fir = []
Yprom = []
Yprom_iir = []
Yprom_fir = []
Zprom = []
Zprom_iir = []
Zprom_fir = []


for gesture_name in classmap:                           # Se recorre cada gesto
    fig, axes = plt.subplots(3,3, figsize=(20, 20))   
    for capture in range(int(len(x))):                  # Se recorre cada renglón de las matrices
        Total = 1
        if (x[capture, N] == gesture_name):             # Si en el último elemento se detecta la etiqueta correspondiente
            # Se grafica la señal en los tres ejes
            axes[0][0].plot(t_resampled, signal.decimate(x[capture, 0:N], decimate_factor) ,color='gray',alpha=0.5)
            Xprom.append(x[capture, 0:N])
            axes[1][0].plot(t_resampled, signal.decimate(y[capture, 0:N], decimate_factor), color='gray',alpha=0.5)
            Yprom.append(y[capture, 0:N])
            axes[2][0].plot(t_resampled, signal.decimate(z[capture, 0:N], decimate_factor) ,color='gray',alpha=0.5)
            Zprom.append(z[capture, 0:N])
            # Se aplica el filtrado sobre la señal
            # N del filtro FIR
            Nfir = len(polos_fir)
            # Se agregan ceros al final de la señal
            senial_x = np.concatenate([signal.decimate(x[capture, 0:N], decimate_factor), np.zeros(round(Nfir/2))])
            senial_x_fir = signal.lfilter(Num_fir, Den_fir, senial_x)
            # Se recortan las primeras N/2 muestras de la señal FIR
            senial_x_fir = senial_x_fir[round(Nfir/2):len(senial_x_fir)]
            Xprom_fir.append(senial_x_fir)
            
            senial_y = np.concatenate([signal.decimate(y[capture, 0:N], decimate_factor), np.zeros(round(Nfir/2))])
            senial_y_fir = signal.lfilter(Num_fir, Den_fir, senial_y)
            # Se recortan las primeras N/2 muestras de la señal FIR
            senial_y_fir = senial_y_fir[round(Nfir/2):len(senial_y_fir)]
            Yprom_fir.append(senial_y_fir)
            
            senial_z = np.concatenate([signal.decimate(z[capture, 0:N], decimate_factor), np.zeros(round(Nfir/2))])
            senial_z_fir = signal.lfilter(Num_fir, Den_fir, senial_z)
            # Se recortan las primeras N/2 muestras de la señal FIR
            senial_z_fir = senial_z_fir[round(Nfir/2):len(senial_z_fir)]
            Zprom_fir.append(senial_z_fir)
            
            axes[0][2].plot(t_resampled, senial_x_fir,color='gray',alpha=0.5)
            axes[1][2].plot(t_resampled, senial_y_fir, color='gray',alpha=0.5)
            axes[2][2].plot(t_resampled, senial_z_fir, color='gray',alpha=0.5)



            senial_x_iir = signal.lfilter(Num_iir, Den_iir,  signal.decimate(x[capture, 0:N], decimate_factor))
            Xprom_iir.append(senial_x_iir)
            
            senial_y_iir = signal.lfilter(Num_iir, Den_iir,  signal.decimate(y[capture, 0:N], decimate_factor))
            Yprom_iir.append(senial_y_iir)

            senial_z_iir = signal.lfilter(Num_iir, Den_iir,  signal.decimate(z[capture, 0:N], decimate_factor))
            Zprom_iir.append(senial_z_iir)

            axes[0][1].plot(t_resampled, senial_x_iir,color='gray',alpha=0.5)
            axes[1][1].plot(t_resampled, senial_y_iir, color='gray',alpha=0.5)
            axes[2][1].plot(t_resampled, senial_z_iir, color='gray',alpha=0.5)

            trial_num = trial_num + 1

    xa = np.mean(Xprom,axis=0)
    axes[0][0].plot(t_resampled, signal.decimate(xa, decimate_factor), linewidth=3,color='blue',label="Promedio")
    Xprom = []
    
    xafir = np.mean(Xprom_fir,axis=0)
    axes[0][2].plot(t_resampled, xafir, linewidth=3,color='green',label="Promedio")
    Xprom_fir = []
    
    xaiir = np.mean(Xprom_iir,axis=0)
    axes[0][1].plot(t_resampled, xaiir, linewidth=3,color='red',label="Promedio")
    Xprom_iir = []
    
    ya = np.mean(Yprom,axis=0)
    axes[1][0].plot(t_resampled, signal.decimate(ya, decimate_factor), linewidth=3,color='blue',label="Promedio")
    Yprom = []
    
    yafir = np.mean(Yprom_fir,axis=0)
    axes[1][2].plot(t_resampled, yafir, linewidth=3,color='green',label="Promedio")
    Yprom_fir = []
    
    yaiir = np.mean(Yprom_iir,axis=0)
    axes[1][1].plot(t_resampled, yaiir, linewidth=3,color='red',label="Promedio")
    Yprom_iir = []
    
    za = np.mean(Zprom,axis=0)
    axes[2][0].plot(t_resampled, signal.decimate(za, decimate_factor), linewidth=3,color='blue',label="Promedio")
    Zprom = []
    
    zafir = np.mean(Zprom_fir,axis=0)
    axes[2][2].plot(t_resampled, zafir, linewidth=3,color='green',label="Promedio")
    Zprom_fir = []
    
    zaiir = np.mean(Zprom_iir,axis=0)
    axes[2][1].plot(t_resampled, zaiir, linewidth=3,color='red',label="Promedio")
    Zprom_iir = []
# Se le da formato a los ejes de cada gráfica
    axes[0][0].set_title(classmap[gesture_name] + " (Aceleración X)")
    axes[1][0].set_title(classmap[gesture_name] + " (Aceleración Y)")
    axes[2][0].set_title(classmap[gesture_name] + " (Aceleración Z)")

    axes[0][1].set_title("IIR - " + classmap[gesture_name] + "(Aceleración X)")
    axes[1][1].set_title("IIR - " + classmap[gesture_name] + "(Aceleración Y)")
    axes[2][1].set_title("IIR - " + classmap[gesture_name] + "(Aceleración Z)")

    axes[0][2].set_title("FIR - " + classmap[gesture_name] + "(Aceleración X)")
    axes[1][2].set_title("FIR - " + classmap[gesture_name] + "(Aceleración Y)")
    axes[2][2].set_title("FIR - " + classmap[gesture_name] + "(Aceleración Z)")
    
    for i in range (0,3):
        for j in range(0,3):
            axes[i][j].grid()
            axes[i][j].legend(fontsize=6, loc='upper right');
            axes[i][j].set_xlabel('Tiempo [s]', fontsize=10)
            axes[i][j].set_ylabel('Aceleración [G]', fontsize=10)
    


    
    trial_num = 0
    plt.tight_layout()
    plt.show()


