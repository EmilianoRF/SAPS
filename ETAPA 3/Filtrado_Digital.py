# -*- coding: utf-8 -*-

"""

Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Filtrado Digital:
    En el siguiente script se ejemplifica el proceso de carga de filtros 
    digitales creados con la herramienta pyFDA y el uso de los mismos para el 
    filtrado de señales.

Autor: Albano Peñalva
Fecha: Agosto 2020

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

#%% Lectura del dataset

FS = 500 # Frecuencia de muestre: 500Hz
T = 3    # Tiempo total de cada registro: 2 segundos


folder = 'hechizos' # Carpeta donde se almacenan los .csv

x, y, z, classmap = process_data.process_data(FS, T, folder)

FS_resample = 40
decimate_factor = int(FS/FS_resample) 

#%% Graficación

ts = 1 / FS                     # tiempo de muestreo
N = FS*T                        # número de muestras en cada regsitro
t = np.linspace(0, N * ts, N)   # vector de tiempos
t_resampled = signal.decimate(t, decimate_factor)
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
            axes[gesture_name][0].plot(t_resampled, signal.decimate(x[capture, 0:N], decimate_factor), label="Trial {}".format(trial_num))
            axes[gesture_name][1].plot(t_resampled, signal.decimate(y[capture, 0:N], decimate_factor), label="Trial {}".format(trial_num))
            axes[gesture_name][2].plot(t_resampled, signal.decimate(z[capture, 0:N], decimate_factor), label="Trial {}".format(trial_num))
            trial_num = trial_num + 1

# Se le da formato a los ejes de cada gráfica
    axes[gesture_name][0].set_title(classmap[gesture_name] + " (Aceleración X)")
    axes[gesture_name][0].grid()
    axes[gesture_name][0].legend(fontsize=6, loc='upper right');
    axes[gesture_name][0].set_xlabel('Tiempo [s]', fontsize=10)
    axes[gesture_name][0].set_ylabel('Aceleración [G]', fontsize=10)
    
    axes[gesture_name][1].set_title(classmap[gesture_name] + " (Aceleración Y)")
    axes[gesture_name][1].grid()
    axes[gesture_name][1].legend(fontsize=6, loc='upper right');
    axes[gesture_name][1].set_xlabel('Tiempo [s]', fontsize=10)
    axes[gesture_name][1].set_ylabel('Aceleración [G]', fontsize=10)
    
    axes[gesture_name][2].set_title(classmap[gesture_name] + " (Aceleración Z)")
    axes[gesture_name][2].grid()
    axes[gesture_name][2].legend(fontsize=6, loc='upper right');
    axes[gesture_name][2].set_xlabel('Tiempo [s]', fontsize=10)
    axes[gesture_name][2].set_ylabel('Aceleración [G]', fontsize=10)

plt.tight_layout()
plt.show()

#%% Cálculo y Graficación de la Transformada de Fourier

# A modo de ejemplo se levanta la primer captura del eje Z del Dataset 
senial = signal.decimate(z[0, 0:N], decimate_factor)

# Se crea una gráfica 
fig1, ax1 = plt.subplots(2, 1, figsize=(15, 15), sharex=True)
fig1.suptitle("Señal de aceleración", fontsize=18)

# Se grafica la señal
ax1[0].plot(t_resampled, senial, label='Señal Contaminada')
ax1[0].set_ylabel('Tensión [V]', fontsize=15)
ax1[0].grid()
ax1[0].legend(loc="upper right", fontsize=15)
ax1[0].set_title('Filtrado FIR', fontsize=15)
ax1[1].plot(t_resampled, senial, label='Señal Contaminada')
ax1[1].set_ylabel('Tensión [V]', fontsize=15)
ax1[1].grid()
ax1[1].legend(loc="upper right", fontsize=15)
ax1[1].set_xlabel('Tiempo [s]', fontsize=15)
ax1[1].set_xlim([0, ts*N])
ax1[1].set_title('Filtrado IIR', fontsize=15)


# Se calcula el espectro de la señal contaminada
f, senial_fft_mod = funciones_fft.fft_mag(senial, FS_resample)

# Se crea una gráfica 
fig2, ax2 = plt.subplots(2, 1, figsize=(15, 15), sharex=True)
fig2.suptitle("Señal de aceleración", fontsize=18)

# Se grafica la magnitud del espectro (normalizado)
ax2[0].plot(f, senial_fft_mod/np.max(senial_fft_mod), label='Señal Original')
ax2[0].set_ylabel('Magnitud (normalizada))', fontsize=15)
ax2[0].grid()
ax2[0].legend(loc="upper right", fontsize=15)
ax2[1].plot(f, senial_fft_mod/np.max(senial_fft_mod), label='Señal Original')
ax2[1].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax2[1].grid()
ax2[1].legend(loc="upper right", fontsize=15)
ax2[1].set_ylabel('Magnitud (normalizada))', fontsize=15)
ax2[1].set_xlim([0, FS_resample/2])

#%% Carga de los Filtros 

# Se cargan los archivo generado mediante pyFDA
filtro_fir = np.load('fir_equiriple.npz', allow_pickle=True)
filtro_iir = np.load('iir_buttherworth.npz', allow_pickle=True) 

# filter_parameters.filter_parameters('filtro_iir.npz')

# Se extraen los coeficientes de numerador y denominador
Num_fir, Den_fir = filtro_fir['ba']     
Num_iir, Den_iir = filtro_iir['ba'] 

# Se expresan las funciones de transferencias (H(z))
z = sy.Symbol('z') # Se crea una variable simbólica z
Hz = sy.Symbol('H(z)')

# Numz_fir = 0
# Denz_fir = 0
# for i in range(len(Num_fir)): # Se arma el polinomio del numerador
#     Numz_fir += Num_fir[i] * np.power(z, -i)
# for i in range(len(Den_iir)): # Se arma el polinomio del denominador
#     Denz_fir += Den_fir[i] * np.power(z, -i)
# print("La función de transferencia del Filtro IIR es:")
# print(sy.pretty(sy.Eq(Hz, Numz_fir.evalf(3) / Denz_fir.evalf(3))))
# print("\r\n")

Numz_iir = 0
Denz_iir = 0
for i in range(len(Num_iir)): # Se arma el polinomio del numerador
    Numz_iir += Num_iir[i] * np.power(z, -i)
for i in range(len(Den_iir)): # Se arma el polinomio del denominador
    Denz_iir += Den_iir[i] * np.power(z, -i)
print("La función de transferencia del Filtro IIR es:")
print(sy.pretty(sy.Eq(Hz, Numz_iir.evalf(3) / Denz_iir.evalf(3)))) 
print("\r\n")

#%% Análisis de los Filtros 

# Se calcula la respuesta en frecuencia de los filtros
f_fir, h_fir = signal.freqz(Num_fir, Den_fir, worN=f, fs=FS_resample)
f_iir, h_iir = signal.freqz(Num_iir, Den_iir, worN=f, fs=FS_resample)

# Se grafican las respuestas de los filtros
ax2[0].plot(f_fir, abs(h_fir), label='Filtro FIR', color='orange')
ax2[0].legend(loc="upper right", fontsize=15)
ax2[1].plot(f_iir, abs(h_iir), label='Filtro IIR', color='green')
ax2[1].legend(loc="upper right", fontsize=15)

# Se evalúa la atenuación en las frecuncias de interés (a modo de ejemplo se 
# utilizan las frecuencia de la interferencia y del primer armónico)
_, h1_fir = signal.freqz(Num_fir, Den_fir, worN=[1, 50], fs=FS_resample)
_, h1_iir = signal.freqz(Num_iir, Den_iir, worN=[1, 50], fs=FS_resample)

print("La atenuación del filtro FIR en 1Hz es de {:.2f}dB".format(20*np.log10(abs(h1_fir[0]))))
print("La atenuación del filtro FIR en 50Hz es de {:.2f}dB".format(20*np.log10(abs(h1_fir[1]))))
print("La atenuación del filtro IIR en 1Hz es de {:.2f}dB".format(20*np.log10(abs(h1_iir[0]))))
print("La atenuación del filtro IIR en 50Hz es de {:.2f}dB".format(20*np.log10(abs(h1_iir[1]))))

# Se extraen polos y ceros de los filtros
zeros_fir, polos_fir, k_fir =   filtro_fir['zpk']  
zeros_iir, polos_iir, k_iir =   filtro_iir['zpk']

# Se grafican las distribuciones de ceros y polos
fig3, ax3 = plt.subplots(1, 2, figsize=(15, 7))
fig3.suptitle("Distribución de Ceros y Polos en el plano Z", fontsize=18)

ax3[0].set_title('Filtro FIR', fontsize=15)
ax3[0].add_patch(patches.Circle((0,0), radius=1, fill=False, alpha=0.1))
ax3[0].plot(polos_fir.real, polos_fir.imag, 'x', label='Polos', color='red',
            markersize=10, alpha=0.5)
ax3[0].plot(zeros_fir.real, zeros_fir.imag, 'o', label='Ceros', color='none',
            markersize=10, alpha=0.5, markeredgecolor='blue')
lim = 1.2 * np.max([np.max(abs(polos_fir)), np.max(abs(zeros_fir))])
ax3[0].set_xlim(-lim, lim)
ax3[0].set_ylim(-lim, lim)
ax3[0].set_ylabel('Imag(z)', fontsize=15)
ax3[0].set_xlabel('Real(z)', fontsize=15)
ax3[0].grid()
ax3[0].legend(loc="upper right", fontsize=12)

ax3[1].set_title('Filto IIR', fontsize=15)
ax3[1].add_patch(patches.Circle((0,0), radius=1, fill=False, alpha=0.1))
ax3[1].plot(polos_iir.real, polos_iir.imag, 'x', label='Polos', color='red',
            markersize=10, alpha=0.5)
ax3[1].plot(zeros_iir.real, zeros_iir.imag, 'o', label='Ceros', color='none',
            markersize=10, alpha=0.5, markeredgecolor='blue')
lim = 1.2 * np.max([np.max(abs(polos_iir)), np.max(abs(zeros_iir))])
ax3[1].set_xlim(-lim, lim)
ax3[1].set_ylim(-lim, lim)
ax3[1].set_ylabel('Imag(z)', fontsize=15)
ax3[1].set_xlabel('Real(z)', fontsize=15)
ax3[1].grid()
ax3[1].legend(loc="upper right", fontsize=12)

#%% Filtrado de la Señal 

# Se aplica el filtrado sobre la señal
# N del filtro FIR
Nfir = len(polos_fir)
# Se agregan ceros al final de la señal
senial_ = np.concatenate([senial, np.zeros(round(Nfir/2))])

senial_fir = signal.lfilter(Num_fir, Den_fir, senial_)
# Se recortan las primeras N/2 muestras de la señal FIR
senial_fir = senial_fir[round(Nfir/2):len(senial_fir)]


senial_iir = signal.lfilter(Num_iir, Den_iir, senial)

# Se grafican las señales filtradas
ax1[0].plot(t_resampled, senial_fir, label='Señal Filtrada (FIR)', color='red',linewidth=1.5)
ax1[0].legend(loc="upper right", fontsize=15)
ax1[1].plot(t_resampled, senial_iir, label='Señal Filtrada (IIR)', color='purple',linewidth=1.5)
ax1[1].legend(loc="upper right", fontsize=15)

# Se calculan y grafican sus espectros (normalizados)
f1_fir, senial_fir_fft_mod = funciones_fft.fft_mag(senial_fir, FS_resample)
f1_iir, senial_iir_fft_mod = funciones_fft.fft_mag(senial_iir, FS_resample)
ax2[0].plot(f1_fir, senial_fir_fft_mod/np.max(senial_fft_mod), 
            label='Senial Filtrada FIR', color='red',linewidth=1.5)
ax2[0].legend(loc="upper right", fontsize=15)
ax2[1].plot(f1_iir, senial_iir_fft_mod/np.max(senial_fft_mod), 
            label='Senial Filtrada IIR', color='purple',linewidth=1.5)
ax2[1].legend(loc="upper right", fontsize=15)
plt.show()

#%% Evaluar Performance del Filtro 

# Se aplica el filtrado sobre la señal 100 veces y se mide el tiempo requerido
# por el algoritmo
t_start_fir = time()
for i in range(500):
    senial_fir = signal.lfilter(Num_fir, Den_fir, senial)
t_end_fir = time()
t_start_iir = time()
for i in range(500):
    senial_iir = signal.lfilter(Num_iir, Den_iir, senial)
t_end_iir = time()

print("El algoritmo de filtrado FIR toma {:.3f}s".format(t_end_fir - t_start_fir))
print("El algoritmo de filtrado IIR toma {:.3f}s".format(t_end_iir - t_start_iir))















