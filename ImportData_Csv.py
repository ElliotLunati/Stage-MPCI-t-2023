# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 17:36:02 2023

@author: lunati

VERSION 1.0
"""

import numpy as np
import re
import pandas as pd
import decimal
from datetime import datetime
from copy import deepcopy


###############################
###### FONCTIONS ANNEXES ######
###############################


def convert_to_number(word):
    """
    :param word: str \n
    
    :return word: renvoie soit l'entier,soit le float (nombre à virgule) que 
    représente la chaîne de caractère si un de ces types est détecté, 
    sinon on renvoie la chaîne de caractère d'origine
    """
    
    try:
        if "." in word:
            return float(word)
        else:
            return int(word)
    except ValueError:
        return word


def extract_words(string, number_conversion=True):
    """
    :param string: str composé de mots et de ';' \n
    :param number_conversion: booléen pour savoir si on veut appliquer la
    fonction 'convert_to_number'. Il est utile de le fixer à False si on sait
    que tous les caractères dans 'string' sont d'un certain type \n
    
    :return word_list: renvoie une liste contenant les mots détéctés dans la 
    chaîne de caractère. Si un mot est détecté comme int ou float, transforme 
    son type au format associé avec la fonction 'convert_to_number'. Si la 
    chaine de caractère contient ';;', ajoute le caractère vide '' à la liste
    """
    
    word_list = string.split(';')  # Liste des mots détéctés dans 'string'
    
    if number_conversion:
        return [convert_to_number(word) for word in word_list]
    else:
        return word_list


def get_sensor_info(csv_file):
    """
    :param csv_file: fichier .csv dont on veut extraire les informations sur 
    la sonde étudiée comme SENSORID, SAMPLEID, ..., TREAT \n
    
    :return d: renvoie un dictionnaire dans lequel les clés sont les entêtes 
    extraites (ex: SENSORID) auxquelles on associe l'information qu'elles 
    portent (ex: FQBA-002-0642)
    """
     
    # On sélectionne la partie du fichier .csv correspondant aux informations 
    # sur la sonde étudiée
    sensor_info = csv_file.iloc[:1]    
    
    # c pour 'column', on sélectionne seulement aux entêtes des informations
    c = list(sensor_info.columns)[0]   
    
    # r pour 'row', on sélectionne seulement les valeurs associées aux entêtes
    r = sensor_info.values[0][0]       
    
    sensor_info_c = extract_words(c) # On extrait les mots contenus dans les
    sensor_info_r = extract_words(r) # chaînes de caractère 'r' et 'c'
    
    # On crée un dictionnaire dans lequel les clés sont les entêtes extraites
    # et les éléments les valeurs associés aux entêtes.
    # Si l'élément vaut '', alors on met 'Nan' à la place 
    d = {}
    for i in range(len(sensor_info_c)):
        if sensor_info_r[i] == '':
            d[sensor_info_c[i]] = np.nan
        else:
            d[sensor_info_c[i]] = sensor_info_r[i]
    
    return d


def get_sensor_WL(csv_file):
    """
    :param csv_file: fichier .csv dont on veut extraire les longueurs d'onde
    étudiées \n
    
    :return WL_list_int: renvoie une liste contenant les longueurs d'onde 
    étudiées
    """
    
    # On sélectionne la partie du fichier .csv correspondant aux LO
    WL = csv_file.iloc[1].values[0]
    
    # On enlève 'LAMBDA' de la liste
    WL_list_str = extract_words(WL, number_conversion=False)[1:] 
    WL_list_int = list(map(int, WL_list_str))
    
    # On renvoie la liste des longueurs d'ondes étudiées, si un '0' se trouve
    # dans la liste cela veut dire qu'aucune mesure n'est associée à cette LO
    
    return WL_list_int 


def get_sensor_ref(csv_file):
    """
    :param csv_file: fichier .csv dont on veut extraire les mesures de 
    référence \n
    
    :return ref_list: renvoie une liste contenant les mesures de référence si elles 
    existent, sinon renvoie None
    """
    
    # On sélectionne la partie du fichier .csv correspondant aux mesures  
    # de référence
    ref = csv_file.iloc[2].values[0] 
    ref_list = extract_words(ref)[1:] # On enlève 'REFERENCE' de la liste
    
    # Si le premier élément de la liste est '', alors on suppose que les autres
    # éléments de la liste sont de la même forme et on renvoie 'None' pour 
    # indiquer qu'il n'y a pas de référence
    if ref_list[0] == '': 
        return None      
                            
    # Sinon on renvoie directement la liste des valeurs de référence
    return ref_list     


def get_sensor_mesures(csv_file, raw_glitch_filter, \
                       round_method, raw_average, raw_glitch_threshold):
    """
    :param csv_file: fichier .csv dont on veut extraire les mesures \n
    :param glitch: distance maximale entre une mesure et la médiane pour toutes 
    les mesures à la même LO pour tous les scans \n
    :param round_method: méthode d'arrondi que l'on souhaite utiliser \n
    
    :return sensor_mesures: renvoie un dictionnaire pouvant contenir : \n
    - mesures_array, une matrice des mesures d'une rampe \n
    - iglitch, une liste où à chaque mesure on associe '0' si aucun glitch 
    n'a été corrigé, '1' si un glitch a été corrigé \n
    - mes_stdx10, une matrice contenant dont chaque élément est 
    le STD d'une colonne x 10 arrondi à l'entier le plus proche
    """
    
    sensor_mesures = {}
    
    # On sélectionne la partie du fichier .csv correspondant aux mesures
    # Si on détéecte que le fichier .csv contient 46 lignes (sans compter la
    # ligne d'entête), alors on a des mesures de TMS et TMR donc on s'arrête 2
    # lignes avant la fin
    if csv_file.shape[0] == 46:
        mesures = csv_file.iloc[3:-2].values
    
    # Sinon on va jusqu'à la fin du fichier 
    else:
        mesures = csv_file.iloc[3:].values
    
    # On enlève "num scan" au début des listes
    # 1 ligne de 'mesures_list' correspond à 1 scan
    mesures_list = [extract_words(scan[0])[1:] for scan in mesures]
    
    # On transforme la liste en matrice numpy
    mesures_array = np.array(mesures_list)
    
    # Filtre les glitchs si nécessaire
    if raw_glitch_filter:
        iglitch = []
        medians = np.median(mesures_array, axis=0)
    
        # On vérifie si tous les éléments d'une des colonnes de 'mesures_array' 
        # se trouvent dans  l'intervalle de glitch
        masks = np.abs(mesures_array - medians) > raw_glitch_threshold
        
        # Si on détecte un glitch, on change la valeur associée par la médiane,
        # sinon on change rien
        mesures_array = np.where(masks, medians, mesures_array)
        
        # On transforme 'masks' en matrice contenant des 0 et des 1 au lieu de
        # True et False
        iglitch = masks.astype(int)
        
    else:
        iglitch = np.zeros(mesures_array.shape)
 
    # Fait la moyenne des données si nécessaire
    if raw_average:
        # On compte le nombre de valeurs qui ne sont pas dans l'intervalle
        iglitch = np.sum(iglitch, axis=0)
        
        # Matrice de taille (n, 1), n la taille de la plage de LO étudiées, dont 
        # chaque élément est le STD d'une colonne x 10
        std_mesures_array = np.std(mesures_array, axis=0, ddof=1)
    
        # On arrondit les éléments de la matrice à l'entier le plus proche
        if round_method == 'Banker':
            mes_stdx10 = np.rint(10 * std_mesures_array).astype(int)
    
        elif round_method == 'RoundUp':
            decimal.getcontext().rounding = decimal.ROUND_HALF_UP
            vectorized_to_integral_value = np.vectorize(lambda x: int(decimal.Decimal(str(x)).to_integral_value()))
            mes_stdx10 = vectorized_to_integral_value(10 * std_mesures_array).astype(int)
    
        else:
            raise ValueError('Warning !', round_method, 'rounding method doesn\'t exist or isn\'t implemented')
    
        # Matrice de taille (n, 1), n la taille de la plage de LO étudiées, dont 
        # chaque élément est la moyenne d'une colonne
        mesures_array = np.mean(mesures_array, axis=0) 
        
        sensor_mesures['mes_stdx10'] = mes_stdx10
    
    sensor_mesures['mesures_array'] = mesures_array
    sensor_mesures['iglitch'] = iglitch
    
    return sensor_mesures


def get_sensor_TMS(csv_file):
    """
    :param csv_file: fichier .csv dont on veut extraire les TMS \n
    
    :return TMS_list: renvoie une liste contenant les TMS
    """
    
    # On sélectionne la partie du fichier .csv correspondant aux valeurs de TMS
    # si elle existe
    if csv_file.shape[0] == 46:
        TMS = csv_file.iloc[-2:].values[-2:-1][0]
        TMS_list = extract_words(TMS[0])[1:] # On enlève 'TMS' de la liste
        return TMS_list
    
    # Sinon on renvoie 'None' pour indiquer qu'il n'y en a pas
    return None


def get_sensor_TMR(csv_file):
    """
    :param csv_file: fichier .csv dont on veut extraire les TMR \n
    
    :return TMR_list: renvoie une liste contenant les TMR
    """
    
    # On sélectionne la partie du fichier .csv correspondant aux valeurs de TMR
    # si elle existe
    if csv_file.shape[0] == 46:
        TMR = csv_file.iloc[-2:].values[-1]
        TMR_list = extract_words(TMR[0])[1:] # On enlève 'TMR' de la liste
        return TMR_list
    
    # Sinon on renvoie 'None' pour indiquer qu'il n'y en a pas
    return None


def get_sensor_name(file_name):
    """
    :param file_name: nom d'un fichier .csv dont on veut connaître le nom de
    la sonde étudiée (ou chemin complet vers le fichier) \n
    
    :return sensor: renvoie une chaîne de caractère contenant le nom de 
    la sonde étduiée 
    """
    
    # Recherche du motif 'FQBx-xxx-xxxx'
    match_fqbx = re.findall(r'FQB\w-\d{3}-\d{3,4}', file_name)
    if match_fqbx:
        fqbx_code = match_fqbx[0]
    else:
        fqbx_code = ''
    
    return fqbx_code


def get_prodname(file_name):
    """
    :param file_name: nom d'un fichier .csv dont on veut connaître la nature
    du produit étudié (ou chemin complet vers le fichier) \n
    
    :return prod_name: renvoie une chaîne de caractère contenant le nom du 
    produit étduié 
    """
    
    # file name est du type : FQBx-xxx-xxxx(' ' ou _)prodname(. ou ' ') ...
    # On enlève donc FQBx-xxx-xxxx et on lit la nouvelle chaîne de caractère
    # (en sautant le premier caractère) jusqu'à ce que l'on voit ' ' ou .
    pattern = r"FQB\w-\d{3}-\d{3,4}[_\s](\w+)[\s.]"
    result = re.search(pattern, file_name)
    prodname = result.group(1)
    
    return prodname 


#################################
###### FONCTION PRINCIPALE ######
#################################
     

def ImportData_Csv(file_path, preproc):
    """
    :param file_path: chemin vers le fichier .csv contenant les données d'une rampe \n
    :param preproc: dictionnaire contenant les différentes procédures à appliquer
    sur les données importées (initialisé dans le fichier .py 'ImportData_ProdFQBA'\n
    
    :return csv_data: renvoie un dictionnaire contenant les données extraites
    """
    
    """ VALIDATION DE PREPROC """
    
    not_air = preproc['prod'][2]
    Master_BDD = deepcopy(preproc['prod'][1])
    raw_glitch_filter = preproc['raw_glitch_filter']
    raw_glitch_threshold = preproc['raw_glitch_threshold']
    raw_resampling = preproc['raw_resampling']
    raw_average = preproc['raw_average']
    round_method = preproc['round_method']
    bkg_proc = preproc['bkg_proc']
    
    """ INITIALISATION DES INFOS A RECUPERER """
    
    # On initialise le nom des informations contenues dans la Master BDD si le
    # produit n'est pas de l'air
    if not_air:
        csv_info_name = ['Type1', 'Type2', 'Type3', 'Project', 'Supplier', \
                         'Country', 'Aspect']
        not_used = ['Origin', 'Description', 'Recipe', 'Supplier reference', \
                    'Quantities', 'Blend recipe', 'Component', 'Value', 'Date', \
                    'Sampling Date', 'Receipt Date', 'HCP values']
        
        index_list = list(Master_BDD.index.values)
        for elem in csv_info_name + not_used:
            index_list.remove(elem) 
        
        csv_prop_name = index_list
    
    # On initialise les variables associées aux données contenues dans le 
    # répertoire 
    csv_sensor_info = {}
    csv_sensor_ref = []
    csv_sensor_mesures = []
    csv_sensor_TMS = []
    csv_sensor_TMR = []
    csv_sensor_iglitch = []
    csv_sensor_filename = []
    csv_sensore_mes_stdx10 = []
    csv_prodname = []
    csv_sensor_name = []
    csv_iScan = []
    csv_time = []
    csv_info = []
    csv_prop = []
    csv_processing_list = []
    
    """ RECUPERATION DES INFOS """
    
    # On lit le fichier .csv
    csv_file = pd.read_csv(file_path)
    
    # Liste contenant la plage de longueurs d'onde étudiée 
    csv_sensor_WL = np.array(get_sensor_WL(csv_file))
    WL_mask = csv_sensor_WL != 0
    folder_sensor_WL = csv_sensor_WL[WL_mask]
    
    # On garde en mémoire la taille de la plage de LO étudiée
    nb_WL = folder_sensor_WL.shape[0]

    # On initialise un vecteur de Nan avant de parcourir toutes les acquis
    # pour ne pas le recalculer à chaque itération
    nan_line = nb_WL * [np.nan]
    
    # Dataframe pandas contenant les informations sur la sonde étudiée 
    # comme SENSORID, SAMPLEID, ..., TREAT et leur valeur associée
    sensor_info = get_sensor_info(csv_file)
        
    # Si 'raw_glitch_threshold' vaut 'prod' on prend la valeur du 
    # fichier .csv
    if raw_glitch_filter == 'prod':
        raw_glitch_threshold = sensor_info['GLITCH']
        if raw_glitch_threshold == '':
            raw_glitch_threshold = 1000000
        
    # Matrice contenant les mesures moyennées d'un scan, le nombre de 
    # glitch pour chaque scan de chaque mesure et le std x10 arrondi à 
    # l'entier supérieur
    sensor_mesures = get_sensor_mesures(csv_file, raw_glitch_filter, \
                     round_method, raw_average, raw_glitch_threshold)
        
    # On ajoute mesures, stdx10, iglitch 
    # On retire les valeurs aux positions où la LO vaut 0 (qui signifie
    # qu'aucune valeur n'a été mesurée)
    iglitch = sensor_mesures['iglitch'][WL_mask]
        
    # Moyenne les données si demandé
    if raw_average:
        csv_sensore_mes_stdx10.append(sensor_mesures['mes_stdx10'][WL_mask])
        csv_sensor_mesures.append(sensor_mesures['mesures_array'][WL_mask])
        # Données moyennées -> iScan vaut 1 pour chaque acquisition (par 
        # défault)
        nb_scan = 1
                   
    else:
        nb_scan = sensor_mesures['mesures_array'].shape[0]
        [csv_sensor_mesures.append(scan[WL_mask]) for scan in sensor_mesures['mesures_array']]
        
    # On vérifie si 'csv_sensor_info'est bien initialisé, ie que les clés
    # du dictionnaire 'sensor_info' se trouvent bien dans 'csv_sensor_info'
    # On multiplie le nombre des éléments dans 'sensor_info' par le nombre
    # de scans si besoin
    if csv_sensor_info:
        for key, val in sensor_info.items():
            new_val = nb_scan * [val]
            csv_sensor_info[key] += new_val
    else:
        for key, val in sensor_info.items():
            csv_sensor_info[key] = nb_scan * [val]
        
    # Matrice contenant les mesures de référence 
    # OU des Nan si le .csv n'en a pas
    sensor_ref = get_sensor_ref(csv_file)
    
    # Liste contenant les TMS
    # OU des Nan si le .csv n'en a pas
    sensor_TMS = get_sensor_TMS(csv_file)
    
    # Liste contenant les TMR
    # OU des Nan si le .csv n'en a pas
    sensor_TMR = get_sensor_TMR(csv_file)

    # Nom du produit étudié
    prod = get_prodname(file_path)
    
    # Nom de la sonde étudiée
    sensor_name = get_sensor_name(file_path)
        
    # On note l'heure à laquelle l'acquisition a été faite (en jour depuis
    # l'an 0), ATTENTION dans BDD TRANSFER on a des fichiers 'Dark' et 
    # 'Reference' n'ont pas de 'DATESMPL'           
    dark_time_str = sensor_info['DATEDARK']
    ref_time_str = sensor_info['DATEREF']
    acqui_time_str = sensor_info['DATESMPL']
        
    if prod == 'Dark':
        acqui_time_str = dark_time_str
    elif prod == 'Reference':
        acqui_time_str = ref_time_str
    
    format_str = '%d/%m/%Y %H:%M:%S'
    acqui_time = datetime.strptime(acqui_time_str, format_str)
    y_0 = datetime(1, 1, 1)
    diff = acqui_time - y_0
    nb_sec = diff.total_seconds()
    acqui_time = nb_sec / 86400
        
    # On extrait de la Master BDD les informations qui nous intéressent
    # en fonction du carburant étudié
    if not_air:        
        # Cas dans BDD_TRANSFER où le fichier étudié correspond soit à une 
        # mesure de Dark, soit de Référence
        # Dans ce cas on laisse Nan 
        if prod == 'Reference' or prod == 'Dark':
            not_air = False
            
        # Sinon on va chercher les données qui nous intéressent 
        else:
            info_list = []
            prop_list = []
    
            for info_name in csv_info_name:
                info = Master_BDD.at[info_name, prod]
                info_list.append(info)
            
            for prop_name in csv_prop_name:
                prop = Master_BDD.at[prop_name, prod]
                prop_list.append(prop)
            
    # Liste des prétraintements déjà appliqués sur les données 
    # spectrales
    processing_list = []
    treat_val = sensor_info['TREAT']
    if treat_val != 0:
        pretreat = ['glitch_filter', 'average', 'dark_model_sub',\
                    'center_scale_compens', 'Tre_corr_HDW', \
                    'Tre_corr_Prod', 'calc_absorb', 'Shift_LO',
                    'Asserv_source']
            
        [pretreat.append('not_used') for i in range(16 - len(pretreat))]
        pretreat = np.array(pretreat)
            
        # Retrouve les prétraitements activés 
        get_bin = lambda x, n: format(x, 'b').zfill(n)
        iactive_treat = get_bin(treat_val, 16)[::-1]
        pretreat_mask = np.array([int(char) for char in iactive_treat])
        pretreat_mask = pretreat_mask == 1
        used_pretreat = pretreat[pretreat_mask]
        for pretreat in used_pretreat:
            # On ne met pas 'Shift_LO' dans 'processing_list' (car pas 
            # possible de le refaire sous Python / Matlab s'il n'a pas
            # été fait lors de l'acquisition)
            if pretreat != 'Shift_LO':
                processing_list.append([[pretreat], np.array([])])
    
    # On ajoute autant de fois qu'il y a de scan les données précédemment
    # extraites dans les listes contenant les données de toutes les acquis
    for i in range(nb_scan):
        csv_sensor_filename.append(file_path)
        csv_prodname.append(prod)
        csv_sensor_name.append(sensor_name)
        csv_sensor_iglitch.append(iglitch)
        csv_time.append(acqui_time)
        csv_iScan.append(i+1)
           
        if not_air:
            csv_info.append(info_list)
            csv_prop.append(prop_list)
            
        if not (sensor_ref is None):   
            csv_sensor_ref.append(sensor_ref)
        else:
            csv_sensor_ref.append(nan_line)
        
        if not (sensor_TMS is None):   
            csv_sensor_TMS.append(sensor_TMS)
        else:
            csv_sensor_TMS.append(nan_line)
        
        if not (sensor_TMR is None):   
            csv_sensor_TMR.append(sensor_TMR)
        else:
            csv_sensor_TMR.append(nan_line)
            
        # Si 'processing_list' n'est pas vide on ajoute les données à 
        # 'csv_processing_list'
        if processing_list:
            csv_processing_list.append(processing_list)
         
    # On crée un dictionnaire pour centraliser les données extraites, pour les
    # plages de longeurs d'ondes on n'en garde qu'une seule car on suppose que
    # dans un répertoire la plage de LO est toujours la même
    csv_data = {'wave' : np.array(csv_sensor_WL),  
                'mesure' : np.array(csv_sensor_mesures),
                'iglitch' : np.array(csv_sensor_iglitch),
                'VMEMS_SET' : np.array(csv_sensor_TMS),
                'VMEMS_MES' : np.array(csv_sensor_TMR),
                'filename' : np.array(csv_sensor_filename),
                'prodname' : np.array(csv_prodname),
                'sensor' : np.array(csv_sensor_name),
                'iScan' : np.array(csv_iScan),
                'time' : np.array(csv_time),
                'processing_list' : np.array(csv_processing_list, dtype=object)} 
                
    # Si le produit n'est pas de l'air on ajoute les informations de la Master
    # BDD associées au produit étudié
    if not_air:
        csv_data['info_name'] = np.array(csv_info_name)
        csv_data['info'] = np.array(csv_info)
        csv_data['prop_name'] = np.array(csv_prop_name)
        csv_data['prop'] = np.array(csv_prop)
                
    if raw_average:
        csv_data['MES_STDx10'] = np.array(csv_sensore_mes_stdx10)
    
    # On parcourt les dictionnaires contenus dans 'csv_sensor_info' pour placer
    # les informations dans 'csv_data'
    for key, val in csv_sensor_info.items():
        csv_data[key] = np.array(val)
          
    # Sauve les données relatives à la référence dans 'csv_bkg'
    csv_bkg = deepcopy(csv_data)
    if 'MES_STDx10' in csv_bkg:
        csv_bkg['MES_STDx10'] = np.full(csv_bkg['MES_STDx10'].shape, np.nan)
    
    csv_bkg['iglitch'] = np.zeros(csv_bkg['iglitch'].shape)
    csv_bkg['mesures'] = csv_sensor_ref
    csv_bkg['processing_list'] = np.empty((1,2), dtype=object)
    csv_bkg['time'] = ref_time_str
     
    return csv_data, csv_bkg
