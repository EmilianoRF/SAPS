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
import numpy as np
import matplotlib.pyplot as plt
from import_ltspice import import_AC_LTSpice
from Diseño_Filtros_Analogicos import SeccOrden2


plt.close('all') # cerrar gráficas anteriores
VPP_entrada = 1.92
Frecuencias_medidas = [1,5,10,15,20,25,30,35,40,45,50,60,80,100,150]
Amplitudes_medidas  = [1.79,1.8,1.7,1.38,0.93,0.636,0.468,0.352,0.276,0.220,0.184,0.120,0.08,0.047,0.028]
Ganancias_medidas   = [Amplitud/VPP_entrada for Amplitud in Amplitudes_medidas]
# %% Se recuperan las funciones de transferencia de las secciones de orden 2
# calculadas en el script anterior

num_1, den_1 = SeccOrden2()

# Se calcula la respuesta en frecuncia de ambas secciones
f = np.logspace(0, 5, int(1e3))
_, h_1 = signal.freqs(num_1, den_1, worN=2*np.pi*f)

# %% Cálculo de componentes Pasa Bajos 1 implementado con Sallen-Key

# Siguiendo el "Mini Tutorial Sallen-Key"

# Se propone el valor de C1
C1_1 = 220e-9 # 220nF

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
R1_1_c = 68.2e3
R2_1_c = 68.2e3
R3_1_c = 0
R4_1_c = np.inf
C1_1_c = C1_1
C2_1_c = 110e-9
print('Los componentes comerciales para la sección 1 son:')
print('R1: {:.2e} Ω'.format(R1_1_c))
print('R2: {:.2e} Ω'.format(R2_1_c))
print('R3: {:.2e} Ω'.format(R3_1_c))
print('R4: {:.2e} Ω'.format(R4_1_c))
print('C1: {:.2e} F'.format(C1_1_c))
print('C2: {:.2e} F'.format(C2_1_c))
print('\r\n')


# %% Comparación de las respuestas en frecuncia de los filtros simulados con
# componentes comerciales

# Luego de simular los filtros utilizando el software LTSpice, se cargan los
# resultados obtenidos para realizar la comparación con el diseño "ideal"

f1, h_sim_1, _ = import_AC_LTSpice('SallenKey_PasaBajos_2.txt')

# Se crea una gráfica para comparar los filtros 
fig, ax = plt.subplots(figsize=(20, 20))

ax.set_title('Sección 1', fontsize=18)
ax.set_xlabel('Frecuencia [Hz]', fontsize=15)
ax.set_ylabel('|H(jw)|² [dB]', fontsize=15)
ax.set_xscale('log')
ax.set_yticks(np.arange(-50,10,5))
ax.set_ylim([-50,5])

ax.set_xlim(1, 1e5)
ax.grid(True, which="both")
ax.plot(f, 20*np.log10((abs(h_1))), label='Ideal')
ax.plot(f1, h_sim_1, label='Simulado')
ax.plot(Frecuencias_medidas, 20*np.log10(Ganancias_medidas), label='Implementado')
ax.scatter(Frecuencias_medidas, 20*np.log10(Ganancias_medidas),color='k',zorder=10)
ax.plot(50, -14.3, 'D', label='Requisito de atenuación',zorder=10)
ax.legend(loc="lower left", fontsize=15)
ax.set_xlim(1,200)



