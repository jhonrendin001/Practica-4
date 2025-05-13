# Practica-4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

# === FUNCIONES DE FILTRADO Y PROMEDIADO ===

def moving_average(data, window=3):
    return data.rolling(window=window, center=True).mean()

def lowpass_filter(data, cutoff=0.1, fs=1/5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def bandpass_filter(data, lowcut, highcut, fs=1/5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data)

def plot_fft(signal, fs=1/5, label=''):
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1/fs)[:N//2]
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]), label=label)

# === CARGA DE DATOS ===

temp = pd.read_csv('temperatura.csv')
hum = pd.read_csv('humedad.csv')
wind = pd.read_csv('viento.csv')

time = np.arange(len(temp)) * 5  # intervalo de 5s

# === GRAFICAR SEÑALES ORIGINALES ===

plt.figure(figsize=(12, 6))
plt.plot(time, temp['valor'], label='Temperatura')
plt.plot(time, hum['valor'], label='Humedad')
plt.plot(time, wind['valor'], label='Viento')
plt.title('Señales Originales')
plt.xlabel('Tiempo (s)')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.show()

# === PROMEDIADO MÓVIL ===

temp_avg = moving_average(temp['valor'])
hum_avg = moving_average(hum['valor'])
wind_avg = moving_average(wind['valor'])

# Graficar temperatura promediada
plt.figure(figsize=(12, 6))
plt.plot(time, temp['valor'], label='Original')
plt.plot(time, temp_avg, label='Promediado')
plt.title('Temperatura - Promediado Móvil')
plt.xlabel('Tiempo (s)')
plt.ylabel('Valor')
plt.legend()
plt.grid()
plt.show()

# === FILTRO PASA BAJAS ===

temp_lp = lowpass_filter(temp['valor'])
hum_lp = lowpass_filter(hum['valor'])
wind_lp = lowpass_filter(wind['valor'])

plt.figure(figsize=(12, 6))
plt.plot(time, temp['valor'], label='Original')
plt.plot(time, temp_lp, label='Pasa Bajas')
plt.title('Temperatura - Filtro Pasa Bajas')
plt.xlabel('Tiempo (s)')
plt.ylabel('Valor')
plt.legend()
plt.grid()
plt.show()

# === FILTRO PASA BANDAS ===

temp_bp = bandpass_filter(temp['valor'], 0.01, 0.08)

plt.figure(figsize=(12, 6))
plt.plot(time, temp['valor'], label='Original')
plt.plot(time, temp_bp, label='Pasa Bandas')
plt.title('Temperatura - Filtro Pasa Bandas')
plt.xlabel('Tiempo (s)')
plt.ylabel('Valor')
plt.legend()
plt.grid()
plt.show()

# === FFT (ESPECTRO DE FRECUENCIA) ===

plt.figure(figsize=(12, 6))
plot_fft(temp['valor'], label='Original')
plot_fft(temp_lp, label='Pasa Bajas')
plot_fft(temp_bp, label='Pasa Bandas')
plt.title('FFT - Temperatura')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.legend()
plt.grid()
plt.show()
