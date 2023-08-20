# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 10:17:28 2023

@author: Emiliano Riffel
"""
# Librerías
import process_data
import numpy as np
import matplotlib.pyplot as plt
#from scipy import fft
import funciones_fft as fft

# Lectura del dataset
FS     = 500 # Frecuencia de muestre: 500Hz
T      = 3    # Tiempo total de cada registro: 2 segundos
folder = 'dataset_figuras' # Carpeta donde se almacenan los .csv
x, y, z, classmap = process_data.process_data(FS, T, folder)

# Se limpia la consola y se cierran todas las figuras.
print("\014")
plt.close('all')

# Tiempo de muestreo
ts = 1 / FS         
# Número de muestras en cada regsitro       
N  = FS*T                        
# Vector de tiempos
t  = np.linspace(0, N * ts, N)
# Frecuencia de muestreo propuesta
fm = 40
# Frecuencia de corte
fc = fm/2
# Sensibilidad del sensor para convertir g a V en [mV]
sensibilidad_sensor = 300 
# Delta de voltaje del conversoren [mV]
deltaV = 3.22
# Vectores para almacenar las amplitudes máximas por encima de fc
maximos_x   = []
maximos_y   = []
maximos_z   = []
# Vectores para almacenar las amplitudes en fc
amplitud_mitad_fc_x = []
amplitud_mitad_fc_y = []
amplitud_mitad_fc_z = []


# Se recorren los registros
trial_num = 0
for gesto in classmap:
    fig, axes = plt.subplots(3,1, figsize=(20, 20))  
    # Se recorre cada renglón de las matrices
    for registro in range(int(len(x))):    
    # Si en el último elemento se detecta la etiqueta correspondiente                
        if (x[registro, N] == gesto):                       
            # Se calcula la magnitud de la transformada rápida de Fourier  
            frec,mag_fft_x  = fft.fft_mag(x[registro, 0:N],FS)
            _,mag_fft_y     = fft.fft_mag(y[registro, 0:N],FS)
            _,mag_fft_z     = fft.fft_mag(z[registro, 0:N],FS)
            # Se multiplican las magitnudes por el factor de sensibilidad del sensor
            # para convertir los valores en G a valores en mV.
            mag_fft_x = mag_fft_x*sensibilidad_sensor
            mag_fft_y = mag_fft_y*sensibilidad_sensor
            mag_fft_z = mag_fft_z*sensibilidad_sensor
            
            # Se guardan las magnitudes de los módulos de las FFTs en fc
            amplitud_mitad_fc_x.append((classmap[gesto],(fc,mag_fft_x[np.where(frec == fc)[0][0]])))
            amplitud_mitad_fc_y.append((classmap[gesto],(fc,mag_fft_y[np.where(frec == fc)[0][0]])))
            amplitud_mitad_fc_z.append((classmap[gesto],(fc,mag_fft_z[np.where(frec == fc)[0][0]])))
            
            # Buscamos todos los índices en frecuencia por encima de fc   
            indices        = list(np.where(frec>(fc))[0])
            # Buscamos la posición en frecuencia del máximo de amplitud por encima de fc 
            indice_maximox = np.argmax(mag_fft_x[indices])
            indice_maximoy = np.argmax(mag_fft_y[indices])
            indice_maximoz = np.argmax(mag_fft_z[indices])
            
            # Almacenamos el valor del máximo a partir de la posición encontrada. Los datos se guardan
            # de la forma:
            #                maximo = (Gesto,[frecuencia,valor])
            maximos_x.append((classmap[gesto], (frec[indices[0] + indice_maximox],
                                               mag_fft_x[indices[0]+indice_maximox])))
 
            maximos_y.append((classmap[gesto],(frec[indices[0]+indice_maximoy],
                                               mag_fft_x[indices[0]+indice_maximoy])))
            
            maximos_z.append((classmap[gesto],(frec[indices[0]+indice_maximoz],
                                               mag_fft_x[indices[0]+indice_maximoz])))
            
            # Graficamos todos los espectros
            axes[0].plot(frec,mag_fft_x[0:round(N/2)], label="Trial {}".format(trial_num))
            axes[1].plot(frec,mag_fft_y[0:round(N/2)], label="Trial {}".format(trial_num))
            axes[2].plot(frec,mag_fft_z[0:round(N/2)], label="Trial {}".format(trial_num))
            trial_num = trial_num + 1

            #Se le da formato a los ejes de cada gráfica
            xlim = [0,51]
            axes[0].set_title(classmap[gesto] + " |FFT{Aceleración X}|")
            axes[0].grid(linestyle='dashed')
            axes[0].legend(fontsize=6, loc='upper right');
            axes[0].set_xlabel('Frecuencia [Hz]', fontsize=10)
            axes[0].set_ylabel('Magnitud [mV]', fontsize=10)
            axes[0].set_xlim(xlim)
         
            axes[1].set_title(classmap[gesto] + " |FFT{Aceleración Y}|")
            axes[1].grid(linestyle='dashed')
            axes[1].legend(fontsize=6, loc='upper right');
            axes[1].set_xlabel('Frecuencia [Hz]', fontsize=10)
            axes[1].set_ylabel('Magnitud [mV]', fontsize=10)
            axes[1].set_xlim(xlim)

            axes[2].set_title(classmap[gesto] + " |FFT{Aceleración Z}|")
            axes[2].grid(linestyle='dashed')
            axes[2].legend(fontsize=6, loc='upper right');
            axes[2].set_xlabel('Frecuencia [Hz]', fontsize=10)
            axes[2].set_ylabel('Magnitud [mV]', fontsize=10)
            axes[2].set_xlim(xlim)

    trial_num = 0
    plt.tight_layout()
    plt.show()

# Ahora para cada gesto buscamos el máximo valor en el total de repeticiones y con esto calculamos
# la ganancia del filtro.
for gesto in classmap:
    print("Datos de " +classmap[gesto]+":")
    max_x = max_y = max_z = 0
    max_x_ = max_y_ = max_z_ = 0
    f_x = f_y = f_z = 0
    for i in range(len(maximos_x)):
        if classmap[gesto] == maximos_x[i][0]:
            # Bucamos en los vectores con máximos por encima de fc
            if max_x<maximos_x[i][1][1]:
                max_x = maximos_x[i][1][1]
                f_x   = maximos_x[i][1][0]
            if max_y<maximos_y[i][1][1]:
                max_y = maximos_y[i][1][1]
                f_y   = maximos_y[i][1][0]
            if max_z<maximos_z[i][1][1]:
                max_z = maximos_z[i][1][1]
                f_z   = maximos_z[i][1][0]
            # Buscamos en los vectores con máximos para fc
            if max_x_<amplitud_mitad_fc_x[i][1][1]:
                max_x_ = amplitud_mitad_fc_x[i][1][1]
            if max_y_<amplitud_mitad_fc_y[i][1][1]:
                max_y_ = amplitud_mitad_fc_y[i][1][1]
            if max_z_<amplitud_mitad_fc_z[i][1][1]:
                max_z_ = amplitud_mitad_fc_z[i][1][1]
                
    # Se muestran los resultados
    print("\t Aceleración X:")
    print("\t \t Frecuencia:",fc,"[Hz] -> Amplitud:",max_x_,"[mV] -> Ganancia:",20*np.log10(deltaV/max_x_),"[dB]\n")
    print("\t \t Frecuencia:",f_x ,"[Hz] -> Amplitud:",max_x ,"[mV] -> Ganancia:",20*np.log10(deltaV/max_x), "[dB]\n")
    print("\t Aceleración Y:")
    print("\t \t Frecuencia:",fc,"[Hz] -> Amplitud:",max_y_,"[mV] -> Ganancia:",20*np.log10(deltaV/max_y_),"[dB]\n")
    print("\t \t Frecuencia:",f_y ,"[Hz] -> Amplitud:",max_y ,"[mV] -> Ganancia:",20*np.log10(deltaV/max_y), "[dB]\n")
    print("\t Aceleración Z:")
    print("\t \t Frecuencia:",fc,"[Hz] -> Amplitud:",max_z_,"[mV] -> Ganancia:",20*np.log10(deltaV/max_z_),"[dB]\n")
    print("\t \t Frecuencia:",f_z ,"[Hz] -> Amplitud:",max_z ,"[mV] -> Ganancia:",20*np.log10(deltaV/max_z), "[dB]\n")
            
            
            