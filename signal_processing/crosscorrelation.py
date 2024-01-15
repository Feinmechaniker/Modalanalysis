#!/usr/bin/env python
# coding: utf-8

# # Experimentelle Modalanalyse (EMA) | Prof. J. Grabow
# ## Signalanalyse
# ### Kreuzkorrelationsfunktionen

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 19:13:41 2023

Program history
15.01.2024    V. 1.0    Start


@author: Prof. Jörg Grabow (grabow@amesys.de)
"""
__version__ = '1.0'
__author__ = 'Joe Grabow'


import numpy as np
import matplotlib.pyplot as plt

# Funktion zur Berechnung der Autokorrelationsfunktion
def autocorrelation_function(data):
    N = len(data) 
    acf = []
    for k in range(N):
        sum_term = sum(data[i] * data[i - k] for i in range(k, N))
        acf.append(sum_term / (N - k))
    return acf

# Funktion zur Berechnung der Kreuzkorrelationsfunktion
def crosscorrelation_function(data_1, data_2):
    N = len(data_1) 
    ccf = []
    for k in range(N):
        sum_term = sum(data_1[i] * data_2[i - k] for i in range(k, N))
        ccf.append(sum_term / (N - k))
    return ccf

def gaussimpuls_vector(N, M, sigma):
    x = np.arange(N)
    gauss_impulse = np.exp(-(x - M)**2 / (2 * sigma**2))
    return gauss_impulse

def sinus_function(x, amplitude, frequency, phase):
    return amplitude * np.sin(2 * np.pi * frequency * x + phase)

def generate_white_noise(N):
    return np.random.randn(N) / 2

def plot_time_series_analysis(title_1, title_2, signal, ccf):
    fig, axs = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle(title_1)

    axs[0].plot(x_values, signal, '-', color='cornflowerblue', label=title_1)
    axs[0].legend(loc='upper right')
    axs[0].set_xlabel('Abtastwerte')
    axs[0].set_ylabel('Amplitude')

    axs[1].plot(x_values, ccf, '-', color='cornflowerblue', label=title_2)
    axs[1].legend(loc='upper right')
    axs[1].set_xlabel('Abtastwerte')
    axs[1].set_ylabel('Amplitude')

    # Zeige ein Gitternetz im Hintergrund an
    for ax in axs:
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

    plt.show()

# Testfunktionen erstellen
N = 1000
x_values = np.arange(N)
impulse_1 = gaussimpuls_vector(N, 200, 10)
impulse_2 = gaussimpuls_vector(N, 700, 10)
ccf = crosscorrelation_function(impulse_2, impulse_1) 
noise = generate_white_noise(N)
impulse_2_noise = impulse_2 + noise
ccf_noise = crosscorrelation_function(impulse_2_noise, impulse_1) 

# Funktion berechnen und plotten
plot_time_series_analysis('Gauss-Impulse', 'Kreuzkorrelation', impulse_1+impulse_2, ccf)
plot_time_series_analysis('TX-Impuls', ' RX-Impuls gestört', impulse_1, impulse_2_noise)
plot_time_series_analysis('Kreuzkorrelation ungestört', 'Kreuzkorrelation gestört', ccf, ccf_noise)

