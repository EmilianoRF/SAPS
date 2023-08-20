# -*- coding: utf-8 -*-

"""

Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Filtrado Analógico:
    En el siguiente script se ejemplifica el proceso de diseño de filtros 
    analógicos.

Autor: Albano Peñalva
Fecha: Septiembre 2020

"""

# %% Librerías

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import sympy as sy

plt.close('all') # cerrar gráficas anteriores

# %% Definición de requisitos

APROXIMACION = 'Butterworth'  # 'Chebyshev' o 'Butterworth'
TIPO =  'Pasa Bajo'         # 'Pasa Bajo', 'Pasa Alto' o 'Pasa Banda'
RIPPLE = 3                  # en dB (usado en aprox. de Chebychev)
FP1 = 20                  # Frec. límite para la banda de paso en Hz (frec. de corte en Butterworth, frec. de fin de ripple para Chebyshev)
FS1 = 40                  # Frec. límite para la banda de rechazo en Hz
AT1 = 14                    # Atenuación mínima en dB en la banda de rechazo (a partir de FS1)

if TIPO == 'Pasa Banda':
    FP2 = 2000              # Frec. límite (superior) para la banda de paso en Hz (para pasa banda)
    FS2 = 3000              # Frec. límite (superior) para la banda de rechazo en Hz (para pasa banda)
    AT2 = 50                # Atenuación mínima en dB en la banda de rechazo (a partir de FS2)
    
# %% Normalización de requisitos

# Aplicando la transformación en frecuencia correspondiente, analizamos los 
# requisitos de nuestro filtros respecto al filtro pasa bajops normalizado.

wp1 = 2 * np.pi * FP1   # Convertimos a rad/s
ws1 = 2 * np.pi * FS1
wp2 = 0
ws2 = 0

if TIPO == 'Pasa Bajo':
    ws1_n = ws1 / wp1   # Transformación Pasa Bajos -> Pasa Bajos
    
if TIPO == 'Pasa Alto':
    ws1_n = wp1 / ws1   # Transformación Pasa Bajos -> Pasa Altos
    
if TIPO == 'Pasa Banda':
    wp2 = 2 * np.pi * FP2
    ws2 = 2 * np.pi * FS2
    w0 = np.sqrt(wp1 * wp2) # Frecuencia central en rad/s
    B = wp2 - wp1           # Ancho de banda
    ws1_n = np.abs(ws1 ** 2 - w0 ** 2) / (ws1 * B) # Transformación Pasa Bajos -> Pasa Banda
    ws2_n = np.abs(ws2 ** 2 - w0 ** 2) / (ws2 * B) # Transformación Pasa Bajos -> Pasa Banda

# %% Cálculo del orden óptimo del filtro

w_n = np.logspace(-1, 2, int(10e3))   # vector de frecuencia (en rad/s) para el filtro normalizado
n = 0   # Orden del filtro
at1_n = 0 # Atenuación del filtro normalizado en ws1
at2_n = 0 # Atenuación del filtro normalizado en ws2

if (TIPO == 'Pasa Bajo' or TIPO == 'Pasa Alto'):
    ws2_n = ws1_n
    condicion = lambda at1, at2: at1 < AT1
    
if (TIPO == 'Pasa Banda'):
    condicion = lambda at1, at2: (at1 < AT1) or (at2 < AT2)
    
# Se crea una gráfica para mostrar las distintas iteraciones 
fig1, ax1 = plt.subplots(1, 1, figsize=(12, 12))
ax1.set_title('Filtro '+APROXIMACION+' Normalizado', fontsize=18)
ax1.set_xlabel('Frecuencia [rad/s]', fontsize=15)
ax1.set_ylabel('|H(jw)|² [dB]', fontsize=15)
ax1.set_xscale('log')
ax1.set_xlim(0.1, 100)
ax1.grid(True, which="both")
ax1.plot(ws1_n, -AT1, 'X', label='Requisito de atenuación')
if (TIPO == 'Pasa Banda'):
    ax1.plot(ws2_n, -AT2, 'X', label='Requisito de atenuación')
ax1.legend(loc="lower left", fontsize=15)

# Se repite el cilo aumentando el orden del filtro hasta obtener las 
# atenuaciones requeridas
while condicion(at1_n, at2_n):
    n = n + 1 # Aumentamos el orden
    
    # Se obtienen ceros, polos y ganancia 
    if APROXIMACION == 'Butterworth':
        [z_n, p_n, k_n] = signal.buttap(n) 
    if APROXIMACION == 'Chebyshev':
        [z_n, p_n, k_n] = signal.cheb1ap(n, RIPPLE)
        
    # Se obtiene Numerador y Denominador de la función de transferencia del 
    # filtro normalizado de orden n
    [num_n, den_n] = signal.zpk2tf(z_n, p_n, k_n)
    
    # Se calcula la atenuación en las frecuencias de interés para evaluar 
    # atenuación y en todo w_n para graficación
    _, at_n = signal.freqs(num_n, den_n, worN=[ws1_n, ws2_n])
    at1_n = -20 * np.log10(abs(at_n[0]))
    at2_n = -20 * np.log10(abs(at_n[1]))
    _, h_n = signal.freqs(num_n, den_n, worN=w_n)
    
    # Se grafica la respuesta en frecuencia del filtro normalizado de orden n
    ax1.plot(w_n, 20*np.log10(abs(h_n)), label='Orden {}'.format(n))
    ax1.legend(loc="lower left", fontsize=15)

# %% Desnormalización de la Función de Transferencia

# En num_n y den_n se encuentran los coeficientes de la función de transferencia
# normalizada del orden seleccionado. Ahora debemos aplicar la transformación 
# en frecuencia correspondiente para desnormalizarla.

s = sy.Symbol('s') # Se crea una variable simbólica s
s_n = sy.Symbol('s_n') 

if TIPO == 'Pasa Bajo':
    s_n = s / wp1   # Transformación Pasa Bajos -> Pasa Bajos
    
if TIPO == 'Pasa Alto':
    s_n = wp1 / s   # Transformación Pasa Bajos -> Pasa Altos
    
if TIPO == 'Pasa Banda':
    s_n = (s ** 2 + w0 ** 2) / (s * B) # Transformación Pasa Bajos -> Pasa Banda

# Se aplica la transformación correspondinte y se simplifica la expresión
num_s = num_n
den_s = 0
for i in range(n + 1):
    den_s = den_s + sy.expand(den_n[i] * np.power(s_n, n - i))
Hs = num_s * sy.expand(den_s.as_numer_denom()[1]) / sy.expand(den_s.as_numer_denom()[0])
Hs = sy.factor(Hs)[0]

print("La función de transferencia del Filtro desnormalizado es:\r\n")    
print(sy.pretty(Hs.evalf(3))) 
print("\r\n")

# Se extraen coeficientes del numerador y denominador
num = np.array(sy.poly(Hs.as_numer_denom()[0], s).all_coeffs()).astype(float)
den = np.array(sy.poly(Hs.as_numer_denom()[1], s).all_coeffs()).astype(float)

# Se calcula la atenuación en las frecuencias de interés para evaluar 
# atenuación y en todo f para graficación
f = np.logspace(0, 5, int(1e3))
_, at = signal.freqs(num, den, worN=[ws1, ws2])
at1 = -20 * np.log10(abs(at[0]))
at2 = -20 * np.log10(abs(at[1]))
_, h = signal.freqs(num, den, worN=2*np.pi*f)

print("La atenuación del filtro en {}Hz es de {:.2f}dB".format(FS1, at1))
if (TIPO == 'Pasa Banda'):
    print("La atenuación del filtro en {}Hz es de {:.2f}dB".format(FS2, at2))
print('\r\n')

# Se crea una gráfica para mostrar la respuesta en frecuencia del filtro diseñado
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 12))
ax2.set_title('Filtro '+APROXIMACION+' '+TIPO, fontsize=18)
ax2.set_xlabel('Frecuencia [Hz]', fontsize=15)
ax2.set_ylabel('|H(jw)|² [dB]', fontsize=15)
ax2.set_xscale('log')
ax2.set_xlim(1, 1e5)
ax2.grid(True, which="both")
ax2.plot(FS1, -AT1, 'X', label='Requisito de atenuación')
ax2.plot(f, 20*np.log10((abs(h))), label=APROXIMACION+' orden {}'.format(n))
if (TIPO == 'Pasa Banda'):
    ax2.plot(FS2, -AT2, 'X', label='Requisito de atenuación')
ax2.legend(loc="lower left", fontsize=15)

# %% Factorización en secciones de orden 2

# Con el objetivo de poder implementarlo usando celdas de Sallen Key o 
# Multiples Realimentaciones se separa la función de transferencia en 
# funciones de orden 2

'''
Esta sección del código no está generalizada para distintos tipos de filtros u
órdenes. Sólo se implementa para un filtro pasa bajos de orden 4. Realizar las 
modificaciones necesarias para implementar otros filtros.
'''

# Se obtienen polos y ceros de la función de transferencia
H = signal.TransferFunction(num, den)
ceros = H.to_zpk().zeros
polos = H.to_zpk().poles
gan = H.to_zpk().gain

# Se separan los polos en pares complejos conjugados
polos_1 = polos[0 : 2]
polos_2 = polos[2 : 4]

# Se separan los ceros (como es un pasabajos la variable ceros está vacía)
ceros_1 = [];
ceros_2 = [];

# Se separa la ganancia con el criterio de que el Pasa Bajos 1 tenga ganancia unitaria 
gan_1 = (abs(polos_1[0])) ** 2;
gan_2 = gan / gan_1;

# Se obtinen numerador y denominador de ambos filtros
[num_1, den_1] = signal.zpk2tf(ceros_1, polos_1, gan_1)
[num_2, den_2] = signal.zpk2tf(ceros_2, polos_2, gan_2)

# %% Análisis de las secciones de orden 2

# Se calculan las respuestas en magnitud de ambas secciones de orden 2 y se
# grafican junto a la del filtro de orden 4
_, h_1 = signal.freqs(num_1, den_1, worN=2*np.pi*f)
_, h_2 = signal.freqs(num_2, den_2, worN=2*np.pi*f)
ax2.plot(f, 20*np.log10((abs(h_1))), label='Sección 1')
ax2.plot(f, 20*np.log10((abs(h_2))), label='Sección 2')
ax2.legend(loc="lower left", fontsize=15)

# Se muestran las funciones de transferencia de ambas secciones  de orden 2
n_sec = 2   # Orden de las secciones

# Sección 1
num1_s = 0
den1_s = 0
# Polinomio del Denominador
for i in range(n_sec + 1):
    den1_s = den1_s + sy.expand(den_1[i] * np.power(s, n_sec - i))
# Polinomio del Numerador
n_n = len(num_1)-1 # Orden del numerador   
for i in range(len(num_1)):    
    num1_s = num1_s + sy.expand(num_1[i] * np.power(s, n_n - i))
# Función de Transferencia
H1s = num1_s / den1_s 
print("La función de transferencia de la 1er sección es:\r\n")    
print(sy.pretty(H1s.evalf(3))) 
print("\r\n")

# Sección 2
num2_s = 0
den2_s = 0
# Polinomio del Denominador
for i in range(n_sec + 1):
    den2_s = den2_s + sy.expand(den_2[i] * np.power(s, n_sec - i))
n_n = len(num_2)-1   
# Polinomio del Numerador 
for i in range(len(num_2)):    
    num2_s = num2_s + sy.expand(num_2[i] * np.power(s, n_n - i))
# Función de Transferencia
H2s = num2_s / den2_s 
print("La función de transferencia de la 2da sección es:\r\n")    
print(sy.pretty(H2s.evalf(3))) 
print("\r\n")

# Se grafican Polos y Ceros
fig4, ax4 = plt.subplots(1, 1, figsize=(10, 10))
ax4.set_title('Ceros y polos', fontsize=18)
ax4.set_xlabel('Imag', fontsize=15)
ax4.set_ylabel('Real', fontsize=15)
ax4.plot(np.real(ceros), np.imag(ceros), 'ob')
ax4.plot(np.real(polos), np.imag(polos), 'xr')
for polo in polos:
    ax4.plot([np.real(polo), 0], [np.imag(polo), 0], 'k--', alpha=.5)
ax4.legend(['Ceros', 'Polos'], loc=2)
ax4.grid()
ax4.axhline(y=0, color='k')
ax4.axvline(x=0, color='k')
# ax4.set_xlim(-2e3, 0)

## Se calcula la frecuencia natural de cada par de polos
i = 0
for polo in polos:
    i = i + 1
    if i%2 == 1:
        w = int((np.real(polo)**2+np.imag(polo)**2)**(1/2)/(2*np.pi))
        w_txt = "Frecuencia natural polo: " + str(w) + "Hz"
        ax4.text(np.real(polo)*1.1, np.imag(polo)*1.05, w_txt, ha='center',fontsize=13)
        print("La frecuencia natural del {}º par de polos es {:.2f}Hz".format(int(i/2)+1, w)) 

# %% Función para devolver los coefficientes de las funciones de transferencia 
# de las secciones de orden 2

def SeccOrden2():
    return num_1, den_1, num_2, den_2