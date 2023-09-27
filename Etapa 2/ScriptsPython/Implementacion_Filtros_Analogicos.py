# -*- coding: utf-8 -*-

"""

Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Filtrado Analógico:
    En el siguiente script se ejemplifica el proceso de cálculo de componentes 
    para implementación de filtros analógicos utilizando filtros activos y el 
    análisis de la simulación de los mismos realizada mediante el software 
    LTSpice.

Autor: Albano Peñalva
Fecha: Septiembre 2020

"""

# %% Librerías

from scipy import signal
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from import_ltspice import import_AC_LTSpice
from Diseño_Filtros_Analogicos import SeccOrden1
import funciones_fft

plt.close('all') # cerrar gráficas anteriores
  
# %% Se recuperan las funciones de transferencia de las secciones de orden 2
# calculadas en el script anterior

#num_1, den_1, num_2, den_2 = SeccOrden2()
num_1, den_1 = SeccOrden1() 

# Se calcula la respuesta en frecuncia de ambas secciones
f = np.logspace(0, 5, int(1e3))
_, h_1 = signal.freqs(num_1, den_1, worN=2*np.pi*f)
#_, h_2 = signal.freqs(num_2, den_2, worN=2*np.pi*f)

# %% Cálculo de componentes Pasa Bajos 1 implementado con Sallen-Key

# Siguiendo el "Mini Tutorial Sallen-Key"

# Se propone el valor de C1
C1_1 = 220e-9 # 100nF

w0_1 = np.sqrt(den_1[2])    # El termino independiente es w0^2
alpha_1 = den_1[1] / w0_1   # El termino que acompaña a s es alpha*w0
H_1 = num_1[0] / den_1[2]   # Numerador = H * w0^2

k_1 = w0_1 * C1_1; 
m_1 = (alpha_1 ** 2) / 4 + (H_1 - 1)

# En Sallen-Key no se pueden implementar filtros con ganancia menor que 1,
# por lo tanto si H es menor o igual a uno se implementa un seguidor de 
# tensión (R3 = cable, R4 = no se coloca)

if (H_1 <= 1): 
    R3_1 = 0
    R4_1 = np.inf 
else:
    # Se propone R3
    R3_1 = 1e3  # 1K
    R4_1 = R3_1 / (H_1 - 1)

C2_1 = m_1 * C1_1
R1_1 = 2 / (alpha_1 * k_1)
R2_1 = alpha_1 / (2 * m_1 * k_1)

print('\r\n')
print('Los componentes calculados para la sección 1 son:')
print('R1: {:.2e} Ω'.format(R1_1))
print('R2: {:.2e} Ω'.format(R2_1))
print('R3: {:.2e} Ω'.format(R3_1))
print('R4: {:.2e} Ω'.format(R4_1))
print('C1: {:.2e} F'.format(C1_1))
print('C2: {:.2e} F'.format(C2_1))
print('\r\n')

# Se utilizaran componentes con valores comerciales para la implementación
R1_1_c = 3.9e3
R2_1_c = 3.9e3
R3_1_c = 0
R4_1_c = np.inf
C1_1_c = C1_1
C2_1_c = 1.5e-8
print('Los componentes comerciales para la sección 1 son:')
print('R1: {:.2e} Ω'.format(R1_1_c))
print('R2: {:.2e} Ω'.format(R2_1_c))
print('R3: {:.2e} Ω'.format(R3_1_c))
print('R4: {:.2e} Ω'.format(R4_1_c))
print('C1: {:.2e} F'.format(C1_1_c))
print('C2: {:.2e} F'.format(C2_1_c))
print('\r\n')

# # %% Cálculo de componentes Pasa Bajos 2 implementado con Multiples Realimentaciones

# # Siguiendo el "Mini Tutorial Multiples Realimentaciones"

# # Se propone el valor de C5
# C5_2 = 10e-9 # 10nF

# w0_2 = np.sqrt(den_2[2])    # El termino independiente es w0^2
# alpha_2 = den_2[1] / w0_2   # El termino que acompaña a s es alpha*w0
# H_2 = num_2[0] / den_2[2]   # Numerador = H * w0^2

# k_2 = w0_2 * C5_2; 

# C2_2 = (4 / alpha_2 ** 2) * (H_2 + 1) * C5_2
# R1_2 = alpha_2 / (2 * H_2 * k_2)
# R3_2 = alpha_2 / (2 * (H_2 + 1) * k_2)
# R4_2 = alpha_2 / (2 * k_2)

# print('Los componentes calculados para la sección 2 son:')
# print('R1: {:.2e} Ω'.format(R1_2))
# print('R3: {:.2e} Ω'.format(R3_2))
# print('R4: {:.2e} Ω'.format(R4_2))
# print('C2: {:.2e} F'.format(C2_2))
# print('C5: {:.2e} F'.format(C5_2))
# print('\r\n')

# # Se utilizaran componentes con valores comerciales para la implementación
# R1_2_c = 1.5e4
# R3_2_c = 6.8e3
# R4_2_c = 1.5e4
# C2_2_c = 1e-8
# C5_2_c = C5_2
# print('Los componentes comerciales para la sección 2 son:')
# print('R1: {:.2e} Ω'.format(R1_2_c))
# print('R3: {:.2e} Ω'.format(R3_2_c))
# print('R4: {:.2e} Ω'.format(R4_2_c))
# print('C2: {:.2e} F'.format(C2_2_c))
# print('C5: {:.2e} F'.format(C5_2_c))
# print('\r\n')

# %% Comparación de las respuestas en frecuncia de los filtros simulados con
# componentes comerciales

# Luego de simular los filtros utilizando el software LTSpice, se cargan los
# resultados obtenidos para realizar la comparación con el diseño "ideal"

f1, h_sim_1, _ = import_AC_LTSpice('AF_SallenKey_PasaBajos.txt')
#f2, h_sim_2, _ = import_AC_LTSpice('MFB_PasaBajos.txt')

# Se crea una gráfica para comparar los filtros 
fig3, ax3 = plt.subplots(2, 2, figsize=(12, 12))

ax3[0, 0].set_title('Sección 1', fontsize=18)
ax3[0, 0].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax3[0, 0].set_ylabel('|H(jw)|² [dB]', fontsize=15)
ax3[0, 0].set_xscale('log')
ax3[0, 0].set_xlim(1, 1e5)
ax3[0, 0].grid(True, which="both")
ax3[0, 0].plot(f, 20*np.log10((abs(h_1))), label='Ideal')
ax3[0, 0].plot(f1, h_sim_1, label='Simulado')
ax3[0, 0].legend(loc="lower left", fontsize=15)

# ax3[0, 1].set_title('Sección 2', fontsize=18)
# ax3[0, 1].set_xlabel('Frecuencia [Hz]', fontsize=15)
# ax3[0, 1].set_ylabel('|H(jw)|² [dB]', fontsize=15)
# ax3[0, 1].set_xscale('log')
# ax3[0, 1].set_xlim(1, 1e5)
# ax3[0, 1].grid(True, which="both")
# ax3[0, 1].plot(f, 20*np.log10((abs(h_2))), label='Ideal')
# ax3[0, 1].plot(f2, h_sim_2, label='Simulado')
# ax3[0, 1].legend(loc="lower left", fontsize=15)

gs = ax3[1, 0].get_gridspec()
ax3[1, 0].remove()
ax3[1, 1].remove()
ax3_big = fig3.add_subplot(gs[1, :])
ax3_big.set_title('Filtro orden 2', fontsize=18)
ax3_big.set_xlabel('Frecuencia [Hz]', fontsize=15)
ax3_big.set_ylabel('|H(jw)|² [dB]', fontsize=15)
ax3_big.set_xscale('log')
ax3_big.set_xlim(1, 1e5)
ax3_big.grid(True, which="both")
#ax3_big.plot(f,  20*np.log10((abs(h_1))) + 20*np.log10((abs(h_2))), label='Ideal')
ax3_big.plot(f,  20*np.log10((abs(h_1))), label='Ideal')

ax3_big.plot(f2, h_sim_1 + h_sim_2, label='Simulado')
ax3_big.plot(f2, h_sim_1 , label='Simulado')
ax3_big.legend(loc="lower left", fontsize=15)
plt.tight_layout()

# Análisis de la atenuación del filtro simulado en las frecuencias de interés
FS = 3000
# se calcula la atenuación en el punto mas cercano a la frecuencia de interés
at_1 = h_sim_1[np.argmin(np.abs(f1-3000))] 
#at_2 = h_sim_2[np.argmin(np.abs(f2-3000))] 
print("La atenuación del filtro simulado en {}Hz es de {:.2f}dB".format(FS, at_1))




