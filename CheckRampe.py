# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 10:11:32 2023

@author: lunati

VERSION 1.0
"""

import matplotlib.pyplot as plt
import numpy as np


def onclick(event):
    # Vérifie si le clic gauche de la souris est utilisé
    if event.button == 1:  
        x = event.xdata
        y = event.ydata
        print(f'Coordonnées du point : x={x}, y={y}')


def check_rampe(data):
    """
    :param data: données extraites d'un fichier .mat contenant les données 
    d'un ensemble d'acquisitions (dans la classe 'Viewstruct')\n
    :return: 
    """
    
    NB_ACQUI = 60
    
    # On vérifier tout de même que NB_ACQUI <= au nombre d'acquisitions réel
    size_arrays = data.mesure.shape[0]
    if NB_ACQUI > size_arrays:
        NB_ACQUI = size_arrays
    
    
    """Fig - Multiplot Tre, Vsrc, Isrc vs. acqui n°"""
    
    # On initialise la fonction 'subplot' en précisant le nombre de lignes et 
    # de colonnes (pour tacer plusieurs graphes sur une figure)
    figure, axis = plt.subplots(3, 1, figsize=(6, 8))
    figure.canvas.manager.set_window_title('Tre, Vsrc, Isrc vs. acqui n°') 
    plt.subplots_adjust(hspace=0.5, bottom=0.05, top=0.95)
    
    # Tre
    axis[0].plot(data.T_DCT[:NB_ACQUI+1], color='k', label='MEMS')
    axis[0].plot(data.T_SRC[:NB_ACQUI+1], color='r', label='Source')
    axis[0].plot(data.T_SMPL[:NB_ACQUI+1], color='b', label='Prod')
    plt.setp(axis[0], xlabel='Acqui n°')
    plt.setp(axis[0], ylabel='C°')
    axis[0].set_title('System Tre vs. acqui n°')
    axis[0].legend(loc='upper left')
      
    # VSRC 
    axis[-2].plot(data.VSRC[:NB_ACQUI+1], color='b', label='VSRC')
    plt.setp(axis[-2], xlabel='Acqui n°')
    plt.setp(axis[-2], ylabel='mV')
    axis[-2].set_title('Source voltage vs. acqui n°')
    
    # ISRC
    axis[-1].plot(data.ISRC[:NB_ACQUI+1], color='r', label='ISRC')
    plt.setp(axis[-1], xlabel='Acqui n°')
    plt.setp(axis[-1], ylabel='mA')
    axis[-1].set_title('Source current vs. acqui n°')
    
    
    """Fig - Multiplot Tre, signal average vs. acqui n°"""
    
    # On initialise la fonction 'subplot' en précisant le nombre de lignes et 
    # de colonnes (pour tacer plusieurs graphes sur une figure)
    figure, axis = plt.subplots(3, 1, figsize=(6, 8))
    figure.canvas.manager.set_window_title('Tre, signal average vs. acqui n°') 
    plt.subplots_adjust(hspace=0.5, bottom=0.05, top=0.95)
    
    # Tre
    axis[0].plot(data.T_DCT[:NB_ACQUI+1], color='k', label='MEMS')
    axis[0].plot(data.T_SRC[:NB_ACQUI+1], color='r', label='Source')
    axis[0].plot(data.T_SMPL[:NB_ACQUI+1], color='b', label='Prod')
    plt.setp(axis[0], xlabel='Acqui n°')
    plt.setp(axis[0], ylabel='C°')
    axis[0].set_title('System Tre vs. acqui n°')
    axis[0].legend(loc='upper left')
    
    # A/D count
    axis[-2].plot(np.mean(data.mesure, axis=1)[:NB_ACQUI+1], \
                  color='b', label='A/D count')
    plt.setp(axis[-2], xlabel='Acqui n°')
    plt.setp(axis[-2], ylabel='A/D count')
    axis[-2].set_title('Spectrum average light intensity vs. acqui n°')
    
    # A/D count / sec
    s_avr_grad = np.divide(\
                 np.diff(np.mean(data.mesure, axis=1)[:NB_ACQUI+1], axis=0), 
                 np.diff(data.time[:NB_ACQUI+1]*24*3600))
        
    axis[-1].plot(s_avr_grad, color='b', label='A/D count')
    plt.setp(axis[-1], xlabel='Acqui n°')
    plt.setp(axis[-1], ylabel='A/D count / sec')
    axis[-1].set_title('Average light intensity gradient vs. acqui n°')
    
    
    """Fig - VMEMS_MES minus VMEMS_SET"""
    
    if np.any(data.T_DCT != 0) and \
        np.any(data.VMEMS_MES != 0) and \
        np.any(data.VMEMS_SET != 0):
    
        nb_WL = len(data.VMEMS_MES[0, :])
        
        # Répéter les valeurs de data.T_DCT nb_WL fois
        x = np.tile(data.T_DCT[:NB_ACQUI+1], nb_WL)  
        
        # 'Aplatir' la différence des tableaux (transforme la différence en une
        # liste unidimensionnelle)
        y = (data.VMEMS_MES[:NB_ACQUI+1]-data.VMEMS_SET[:NB_ACQUI+1]).flatten()  
        colors = np.arange(len(y))
        
        plt.figure('VMEMS_MES minus VMEMS_SET vs. T_MEMS')
        plt.scatter(x, y, c=colors, cmap='viridis', s=4)
           
        plt.xlabel('Tre MEMS (°C)')
        plt.ylabel('Delta voltages (V)')
        plt.title('VMEMS_MES minus VMEMS_SET vs. T_MEMS')
        plt.gcf().canvas.mpl_connect('button_press_event', onclick) 
        plt.show()
      
    
    """Fig - Multiplot Tre, stdev vs. acqui n°""" 
    
    # On initialise la fonction 'subplot' en précisant le nombre de lignes et 
    # de colonnes (pour tacer plusieurs graphes sur une figure)
    figure, axis = plt.subplots(2, 1, figsize=(6, 8))
    figure.canvas.manager.set_window_title('Tre, stdev vs. acqui n°') 
    plt.subplots_adjust(hspace=0.5, bottom=0.05, top=0.95)
    
    # Tre
    axis[0].plot(data.T_DCT[:NB_ACQUI+1], color='k', label='MEMS')
    axis[0].plot(data.T_SRC[:NB_ACQUI+1], color='r', label='Source')
    axis[0].plot(data.T_SMPL[:NB_ACQUI+1], color='b', label='Prod')
    plt.setp(axis[0], xlabel='Acqui n°')
    plt.setp(axis[0], ylabel='C°')
    axis[0].set_title('System Tre vs. acqui n°')
    axis[0].legend(loc='upper left')
    
    # STD
    axis[-1].plot(0.1*data.MES_STDx10[:NB_ACQUI+1])
    axis[-1].plot(np.mean(0.1*data.MES_STDx10[:NB_ACQUI+1], axis=1),\
                  c='m', label='STD mean', linewidth=4)
    plt.setp(axis[-1], xlabel='Acqui n°')
    plt.setp(axis[-1], ylabel='Std dev. (LSB)')
    axis[-1].set_title('Stdev calc on scans of each acqui vs. acqui n°')
    axis[-1].legend(loc='upper left')
    
    
    """Fig - Stdev vs. WL index"""
    
    plt.figure('Stdev vs. WL index')
    plt.plot(0.1*data.MES_STDx10[:NB_ACQUI+1].T)
    plt.plot(np.mean(0.1*data.MES_STDx10[:NB_ACQUI+1], axis=0),\
                  c='m', label='STD mean', linewidth=4)
    plt.xlabel('WL index')
    plt.ylabel('Std dev. (LSB))')
    plt.title('Stdev vs. WL index')
    
    
    """Fig - Spectra"""
            
    plt.figure('Spectra')
    plt.plot(data.wave, data.mesure[:NB_ACQUI+1].T)
    plt.xlabel('wave')
    plt.ylabel('A/D count')
    plt.title('light intensity vs. wave')

