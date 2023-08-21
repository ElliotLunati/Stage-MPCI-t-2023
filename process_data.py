# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 14:10:11 2023

@author: lunati

VERSION 1.0
"""

import numpy as np
import re
from copy import deepcopy
from scipy.sparse import spdiags
from scipy.linalg import cho_factor, cho_solve
from RoundUp import RoundUp
from scipy import io
from now import datestr


###############################
###### FONCTIONS ANNEXES ######
###############################


def movmean(x, n):
    """
    :param x: vecteur (1, p) sous la forme d'une matrice numpy \n
    :param n: entier indiquant le nombre de points sur lesquels on veut 
    faire notre moyenne \n
    
    :return m: vecteur (1, p) sous la forme d'une matrice numpy contenant la 
    moyenne sur n points du vecteur de départ
    """
    m = np.cumsum(x) / n
    m[n:] = m[n:] - m[:-n]
    
    for i in range(n+1):
        m[i] = m[i] * n / (i+1)
    
    return m


def wsmoothx(x, lambda_val=1, dorder=1, w=None):
    """
    :param x: vecteur (1,p) sous la forme d'une matrice numpy à lisser \n
    :param lambda_val: méta-paramètre de lissage sous la forme d'un entier 
    (fixé à 1 par défault) \n
    :param dorder: ordre de dérivation utilisé, en général vaut 1, 2 ou 3 
    (fixé à 1 par défault) \n
    :param w: vecteur de poids (1,p) sous la forme d'une matrice numpy, en 
    général est constitué que de 1 - OPTIONNEL \n
    
    :return xs: vecteur (1,p) sous la forme d'une matrice numpy contenant le
    vecteur x lissé
    """
    if w is None:
        w = np.ones_like(x)
    
    if dorder < 1:
        raise ValueError("dorder doit être supérieur ou égal à 1.")
    
    p = len(x)
    # Transposer le vecteur en vecteur colonne si ce n'est pas déjà le cas
    if x.ndim < 2:
        x = x[:, np.newaxis] 
        
    I = np.eye(p)
    D_dense = np.diff(I, n=dorder, axis=0)
    W = spdiags(w, 0, p, p, format='csr')
    C = cho_factor(W + lambda_val * D_dense.T @ D_dense)
    xs = cho_solve(C, w * x)

    return xs[:,0]


def findconsec(x):
    """
    :param x: vecteur (1, p) ou (p, 1) dont sous la forme d'une matrice numpy
    dont nous voulons repérer les séquences de valeurs consécutives
    \n
    
    :return index: renvoie l'index associé aux valeurs consécutives \n
    :return val: renvoie les valeurs associées à l'index'
    """
    
    # Transposer le vecteur en vecteur ligne si ce n'est pas déjà le cas
    if x.ndim > 1:
        x = x.T  

    # calcule la dérivée de x pour trouver les "1" qui correspondent à une 
    # série d'entiers consécutifs
    dx = np.diff(x)
    
    # ajoute NaN au début pour compenser le décalage d'index dû à la dérivée 
    # qui a un élément de moins
    dx = np.concatenate(([np.nan], dx))  
    
    # ajoute également un 0 à la fin pour être sûr d'entourer une série 
    # d'entier consécutifs qui termineraient "x" (e.g. x=[1 2 3])
    dx = np.concatenate((dx, [0]))  

    # remplace tous les éléments différents de 1 par 0
    dx[dx != 1] = 0

    # vérifie qu'il existe au moins une série d'entiers consécutifs (dans le 
    # cas contraire, on sort directement)
    if np.sum(dx) == 0:
        index = []
        val = []
        return index, val

    # on a maintenant des groupes de 1 qui sont entourés par des 0 
    # (e.g. 0 0 0 1 1 1 0 0 1 0),
    # il suffit donc de retrouver ces 0 qui délimitent les séries de 1 pour 
    # connaître les index de ces séries
    izero = np.where(dx == 0)[0]  # retrouve les index de tous les zéros
    diz = np.diff(izero)
    
    # calcule la dérivée de ces index
    diz = np.concatenate(([np.nan], diz))  
    
    # les zéros "délimitant à droite" sont ceux pour lesquels la dérivée de 
    # l'index est supérieure à 1
    izero_dlm_right = izero[np.where(diz > 1)[0]] - 1  
    
    # les zéros "délimitant à gauche" sont ceux qui précèdent directement les 
    # délimitant gauche dans la liste des 0
    izero_dlm_left = izero[np.where(diz > 1)[0] - 1]  

    # on ne cherche pas les index des 0 mais bien des 1, qui sont maintenant 
    # directement identifiables depuis izero_dlm_right & left
    index = []
    val = []
    for i in range(len(izero_dlm_left)):
        index.append(np.arange(izero_dlm_left[i], izero_dlm_right[i] + 1))
        val.append(x[index[i]])

    return index, val 


def findpattern(vector, pattern, elimination=0):
    """
    :param vector: vecteur (1, p) sous la forme d'une matrice numpy dans 
    lequel on veut identifier un pattern \n
    :param pattern: pattern à identifier sous la forme d'un vecteur (matrice 
    numpy (1, p))\n
    :param elimination: vaut 0 ou 1 (0 par défault). Par exemple
    findpattern([1 2 1 2 1 2 1], [1 2 1], 0) donne les indices [0 2 4] 
    findpattern([1 2 1 2 1 2 1], [1 2 1], 1) donne les indices [0 4]
    \n
    
    :return indices: renvoie un vecteur (1, p) sous la forme d'une matrice 
    numpy contenant les indices de la position où le pattern a été détecté
    """
    vector_length = len(vector)
    pattern_length = len(pattern)
    indices = []
    i = 0
    
    while i <= vector_length - pattern_length:
        if np.array_equal(vector[i:i+pattern_length], pattern):
            indices.append(i)
            
            if elimination == 0:
                vector = np.delete(vector, np.arange(i, i+pattern_length))
                vector_length -= pattern_length
        i += 1
    
    return np.array(indices)


#################################
###### FONCTION PRINCIPALE ######
#################################


def process_data(data, proc_list):
    """
    :param data: dictionnaire contenant les données de la sous structure 
    'prod' de la structure 'raw' d'un fichier .mat \n
    :param proc_list: traîtement des données qu'on veut appliquer, en 
    fonction de la requête il peut être de la forme : \n
    - add_tre_grad : process_data(data, ['add_tre_grad', []]) \n
    - apply_query : process_data(data, [['apply_query', query]]) où query 
      est un ensemble de requête du type a >= b ... \n
    - find_RTcycles : process_data(data, ['find_RTcycles', []]) ou 
      process_data(data, ['find_RTcycles', 5e-4]) par exemple si on veut
      préciser un threshold \n
    - find_T_DCT_stable : process_data(data, ['find_T_DCT_stable', []]) ou 
      process_data(data, ['find_T_DCT_stable', 5e-4]) par exemple si on veut
      préciser un threshold \n
    - average / median : process_data(ref_prod, [['average', 'op', round_method]]) ou 
      process_data(ref_prod, [['median', 'op', round_method]]) où op peut être 
      'repet', 'repro' ou 'scan' et round_method correspond à la méthode d'arrondi
      à utiliser (soit Banker, soit RoundUp (Matlab 2013))\n
    - concat : process_data(data, [['concat', [data2, ..., dataN]]]) \n
    - change_subset : process_data(data, [['change_subset', [[subpos1, ..., subposN], 'source->target']]])
      où subpos sont des entiers \n 
      
    
    :return data_proc: renvoie les données sous la forme d'un dictionnaire où
    les traîtements demandés ont été appliqués
    """
    
    data_proc = deepcopy(data)
    processing_list = deepcopy(data['processing_list'].tolist())
    
    # Partie nécessaire car lors de la sauvegarde des fichiers avec io.savemat
    # si 'processing_list' est de dimension (1,2), il le change en (2,) 
    # (pas de solution pour corriger ça à priori malheureusement) qui pose 
    # problème lors de l'ajout de nouveaux process (si on ne le changeait pas
    # 'processing_list' aurait pour dimension (n,) rendant plus compliqué la
    # lisibilité de 'processing_list')
    if not data['processing_list'].shape[0] == 0 and len(data['processing_list'].shape) == 1:
        processing_list = [[processing_list[0], processing_list[1]]]
    
    shared_fields = ['wave', 'prop_name', 'info_name', 'processing_list']
    for proc in proc_list:
        # Met à jour la liste des prétraitements appliqués aux données
        processing_list.append([proc[0], proc[1]])
        
        # Ajoute les gradients de température
        if proc[0] == 'add_tre_grad':
            time = data_proc['time']
            temp_list = ['T_DCT', 'T_SRC', 'T_SMPL', 'T_DCTPCB']
            
            # On ajoute les gradients de température à 'data_proc'
            for temp in temp_list:
                val = data_proc[temp]
                grad_name = temp + '_GRAD'
                t_dct_grad = np.array([0])
                grad = np.divide(np.diff(val, axis=0), np.diff(time*24*3600))
                t_dct_grad = np.append(t_dct_grad, grad)
                data_proc[grad_name] = t_dct_grad
                
        # Applique la requête d'extraction (vecteur de 0/1 ou d'indices absolus
        # pour sélectionner les lignes de données)
        elif proc[0] == 'apply_query':
                  
            # On vérifie que la requête est bien exploitable
            if not np.all(isinstance(element, bool) for element in proc[1]):
                raise ValueError("The query you are trying to apply is not a \
                                 boolean object")
            
            elif len(proc[1]) != len(data_proc['prodname']):
                raise ValueError("Number of elements in the query logical \
                vector does not match the number of acquisition in the data \
                structure ")
                
            else:
                for key, val in data_proc.items():
                    if not key in shared_fields :                                 
                        index_to_keep = proc[1]
                        if len(data_proc[key].shape) == 1:
                            data_proc[key] = data_proc[key][index_to_keep]
                        else:
                            data_proc[key] = data_proc[key][index_to_keep,:]
                    
        
        # On trouve les cycles de température du type 'montée, descente, 
        # montée, descente'
        elif proc[0] == 'find_RTcycles':   
            # Limite sur T_DCT_GRAD pour l'identification des plateaux
            threshold = proc[1]
            len_arrays = len(data_proc['prodname'])
            iRTcycle = np.zeros((len_arrays,))
            if not threshold: 
                threshold = 5*10**-4
                
            # On lisse légèrement le gradient de T_DCT pour faciliter la 
            # détection
            t_dct_grad = wsmoothx(data_proc['T_DCT_GRAD'])
            
            # Utilisation d'un filtre médian pour gommer les baisses ou montées
            # de température anormales lors d'une RT (e.g. RT du 22-03-01 sur
            # ACLF__ FRC02980 FQBI-011)
            n_elts = 5 # à optimiser en fonction du nombre de mesures
            t_dct_grad_medfilt = movmean(t_dct_grad, n_elts)
            
            # On conserve t_dct_grad pour les premiers éléments (début de rampe
            # présente ddes gradients trop aléatoires)
            if len_arrays >= 5*n_elts:
                t_dct_grad_medfilt[:5*n_elts] = t_dct_grad[:5*n_elts]
            else:
                t_dct_grad_medfilt = t_dct_grad
                
            delta = np.abs(t_dct_grad_medfilt - t_dct_grad)
            i2be_corrected = delta > threshold
            t_dct_grad = np.where(i2be_corrected, \
                                  t_dct_grad_medfilt, t_dct_grad)
            
            # Force la première dérivée à 1 au lieu de 0 car les RT commencent 
            # toujours par une montée en température
            t_dct_grad[0] = 1
            
            # Pour éviter les changements de pente dus au bruit, les dérivées
            # trop faibles sont forcées à 0
            lowgrad_corrected = np.abs(t_dct_grad)<=threshold
            t_dct_grad = np.where(lowgrad_corrected, 0, t_dct_grad)
                
            # Garde uniquement le signe (trouve montée, descente, plateau sous
            # forme +1, 0, -1)
            grad_sign = np.sign(t_dct_grad)
            
            # On veut trouver uniquement les segments de T_DCT qui 
            # correspondent à des montées (en incluant le plateau de fin de 
            # montée) et les descentes (en incluant le plateau de fin de 
            # descente)
            # Pour ça on associe à la valeur 0 (plateau), la valeur +1,-1 qui
            # la précède (i.e. plateau en fin de montée passe de 0 à +1 (donc
            # associé à montée) et inversement pour les descentes (0->-1))
            for i in range(1,len_arrays):
                if grad_sign[i] == 0:
                    grad_sign[i] = grad_sign[i-1]
            
            # Retrouve les passages de montées à descente (et inversement) et
            # qui correspondent à un passage de 'grad_sign' de +1 à -1 (resp.
            # -1 +1)
            edges = np.diff(grad_sign)
            
            # Recale la dérivée en ajoutant 0 pour ne pas introduire un 
            # changement de signe non voulu
            edges = np.append([0], edges)
            edges_index = np.where(np.abs(edges) > 1)[0]
            edges_index = np.concatenate(([0], edges_index, [len_arrays]))
            
            # Retrouve les indices des échantillons correspondant aux montées 
            # et aux descentes
            segment_index = []
            segment_sign = []
            for i in range(len(edges_index)-1):
                segment_index.append([edges_index[i], edges_index[i+1]-1])
                segment_sign.append(grad_sign[segment_index[i][1]])
            segment_index = np.array(segment_index)
            segment_sign = np.array(segment_sign)
            
            # Recherche le pattern de type 'montée, descente, montée, descente'
            # de nos RT dans les données de température
            target_pattern = np.array([1, -1, 1, -1])
            idx = findpattern(segment_sign, target_pattern, 1)
            idx = np.column_stack((idx, idx + len(target_pattern) - 1))

            # Associe un numéro de cycle à chaque acquisition
            if idx.size > 0:
                # Cas où on était sur une rampe en température classique pour 
                # laquelle on a retrouvé une ou plusieurs fois le pattern
                # recherché
                for i in range(idx.shape[0]):
                    iRTcycle[segment_index[idx[i, 0], 0]:\
                             segment_index[idx[i, 1], 1] + 1] = i + 1
    
                # La dernière partie des données correspond à un cycle 
                # incomplet auquel on donne un indice négatif, e.g. '-2' (2ème
                # cycle mais inachevé)
                iRTcycle[segment_index[idx[-1, 1], 1] + 1:] \
                    = -(iRTcycle[segment_index[idx[-1, 1], 1]] + 1)
            
            else:
                # CAs où le pattern recherché n'a pu être retrouvé dans les
                # données
                # Attribu par défault la valeur '1' à toutes les acquisitions
                iRTcycle[:] = 1
            
            data_proc['iRTcycle'] = iRTcycle.astype(int)
            
        
        # On trouve et différencie les paliers de température
        elif proc[0] == 'find_T_DCT_stable':
            # Valeur du seuil en-dessous duquel la température est considérée 
            # comme stable
            threshold = proc[1]
            len_arrays = len(data_proc['prodname'])
            if not threshold: 
                threshold = 10**-4
                
            # On initialise la valeur à -1 par défault (i.e. aucune mesure
            # stabilisée en tre)
            T_DCT_STABLE = np.full((len_arrays,), -1)
            
            # On lisse le gradient de température pour ne pas être perturbé par
            # le bruit
            tre_grad_smoothed = wsmoothx(data_proc['T_DCT_GRAD'], 40, 2)
            
            # Les paliers stabilisés en température correspondent aux points 
            # pour lesquels le gradient de température est inférieur au seuil
            # défini
            istable = np.where(np.abs(tre_grad_smoothed) < threshold)[0]
            istable_consec = findconsec(istable)
            
            # Retire les 'paliers' qui correspondent à moins de e.g. 20 
            # acquisitions consécutives 
            istable_consec_index = []
            istable_consec_val = []
            for i in range(len(istable_consec[1])):
                if len(istable_consec[1][i]) >= 20:
                    istable_consec_index.append(istable_consec[0][i])
                    istable_consec_val.append(istable_consec[1][i])
            
            # On attribue un indice à chaque palier où la température est 
            # stable
            for i in range(len(istable_consec_val)):
                T_DCT_STABLE[istable_consec_val[i]] = i + 1
            
            data_proc['T_DCT_stable'] = T_DCT_STABLE
        
        
        # On moyenne / prend la médianne des données (sur les scans, repro ou 
        # repet)
        elif proc[0] == 'average' or proc[0] == 'median':
            # On retrouve la fonction à appliquer sur les données
            func2apply = proc[0]
            
            # On retrouve la méthode d'arrondi à appliquer 
            try:
                round_method = proc[2]
            except:
                raise ValueError('You must specify a rounding method')
            
            round_methods = ['Banker', 'RoundUp']
            if not round_method in round_methods:
                raise ValueError(round_method, 'isn\'t a valid rounding method. Supported methods are : Banker, RoundUp')
            
            # Définition du dictionnaire de correspondance des fonctions
            corr_func = {'average': np.mean,
                         'median': np.median}
                                                   
            # Génère un ID pour chaque ligne
            # le principe est de multiplier par un facteur des informations 
            # descriptives de chaque ligne pour obtenir un identifiant unique
            # e.g. : produit = TOLUENE, iRepro=2, iRepet=3, iScan=42
            #       TOLUENE) -> e.g. 7 -> 7 * xfactor(1) -> e.g. 7 000 000
            #       si on a détecté 6 produits distincts avant 
            #       iRepro -> 2 * xfactor(2) -> e.g. 20 000
            #       iRepet -> 3 * xfactor(3) -> e.g. 3
            #       iScan -> 42 * xfactor(4) -> e.g. 0.0042
            #       scan_ID = 7e6 + 20e4 + 3 +42e-4 = 7 020 003.0042
            # scan_ID est ainsi unique pour chaque combinaison de produit, 
            # repro, repet, scan
            # Ensuite, on peut facilement retrouver les lignes qui ont une ou 
            # plusieurs "propriétés" communes en soustrayant cette propriété
            # a scan_ID
            # e.g. : pour retrouver toutes les lignes qui correspondent aux 
            #        scans du même produit, repro, repet: 
            #        scan_ID - xfactor(4)*proc.iScan -> la partie décimale 
            #        devient nulle et toutes les lignes de même produit, repro, 
            #        repet ont exactement le même scan_ID
        
            # Facteur pour (dans l'ordre): produit, repro, repet, scan
            xfactor = [10**9, 10**5, 1, 10**-5]  
        
            # Obtenir les catégories uniques de data_proc.prodname
            unique_categories = np.unique(data_proc['prodname'])
            
            # Créer une correspondance entre les catégories et les 
            # valeurs numériques
            corr = {categorie: i+1 for i, \
                    categorie in enumerate(unique_categories)}
            
            # Convertir la matrice en matrice de type double avec les valeurs 
            # numériques correspondantes
            corr_array = np.vectorize(corr.get)(data_proc['prodname'])
            
            scan_ID = corr_array * xfactor[0] + \
                      data_proc['iRepro'] * xfactor[1] + \
                      data_proc['iRepet'] * xfactor[2] + \
                      data_proc['iScan'] * xfactor[3]
            
            # Permet de moyenner les scans d'une même acquisition (e.g. passe 
            # de 127 lignes par acquisition à une seule)
            if proc[1] == 'scan':
                temp_ID = scan_ID - xfactor[-1] * data_proc['iScan']
                
                # Nécessaire pour éviter des problèmes d'arrondis qui 
                # engendrent deux nombres quasi identiques 
                # (e.g. 200.0001 et 200.00010000000001)
                if round_method == 'Banker': # Donne possiblement un résultat un peu différent que sur Matlab
                    temp_ID = np.round(temp_ID * 1 / np.min(xfactor)) * np.min(xfactor)
                else:
                    temp_ID = RoundUp(temp_ID * 1 / np.min(xfactor)) * np.min(xfactor)
                
                # Retrouve les lignes correspondant à tous les scans de 
                # chaque acquisition
                unique_ID, irows = np.unique(temp_ID, return_inverse=True)
                
                # Initialise un vecteur de False pour savoir quelles sont 
                # les lignes qui seront conservées par la suite
                ikept = np.full(temp_ID.shape, False)
                
                # Initialise la matrice qui va conserver l'écart-type des 
                # scans de chaque acqui
                if 'MES_STDx10' not in data_proc.keys():
                    data_proc['MES_STDx10'] = np.full_like(data_proc['mesure'], np.nan)
                    
                # Pour chaque groupe, procède à la moyenne des scans pour 
                # la mesure (tous les autres champs d'une même acqui étant 
                # identiques d'un scan à l'autre)
                for i in range(len(unique_ID)):
                    # Sauve la variabilité de la mesure entre scans
                    data_proc['MES_STDx10'][irows==i][0, :] = np.std(data_proc['mesure'][irows==i, :], axis=0)
                        
                    # Moyenne des scans
                    data_proc['mesure'][irows==i][0, :] = corr_func[func2apply](data_proc['mesure'][irows==i, :], axis=0)
        
                    # Conserve un indice du numéro de la ligne qui contient 
                    # les données moyennées et qu'il faut donc conserver 
                    # par la suite
                    ikept[irows==i] = True
            
            
            # Permet de moyenner les répéts d'une même repro (e.g. moyenne les  
            # acquisitions -1, -2, -3 consécutives qui consistuent une repro
            # d'un produit
            elif proc[1] == 'repet':
                temp_ID = scan_ID - (xfactor[-2] * data_proc['iRepet'] + xfactor[-1] * data_proc['iScan'])
                
                # Nécessaire pour éviter des problèmes d'arrondis qui 
                # engendrent deux nombres quasi identiques 
                # (e.g. 200.0001 et 200.00010000000001)
                if round_method == 'Banker':
                    temp_ID = np.round(temp_ID * 1 / np.min(xfactor)) * np.min(xfactor)
                else:
                    temp_ID = RoundUp(temp_ID * 1 / np.min(xfactor)) * np.min(xfactor)

                # Retrouve les lignes correspondant à toutes les répétitions 
                # de chaque repro d'un produit
                unique_ID, irows = np.unique(temp_ID, return_inverse=True)

                # Initialise un vecteur de False pour savoir quelles sont les 
                # lignes qui seront conservées par la suite
                ikept = np.full(temp_ID.shape, False)

                # Pour chaque groupe de répét, procède à la moyenne des données
                for i in range(len(unique_ID)):
                    for ifield in ['mesure', 'T_DCT', 'T_DCTPCB', 'T_SMPL', \
                                   'T_SRC', 'VSRC', 'ISRC', 'DARK', \
                                   'VMEMS_SET', 'VMEMS_MES']:
                        
                        # Moyenne des spectres
                        if len(data_proc[ifield].shape) == 1:
                            data_proc[ifield][np.where(irows==i)[0][0]] = corr_func[func2apply](data_proc[ifield][np.where(irows==i)[0]], axis=0)
                        else:
                            data_proc[ifield][np.where(irows==i)[0][0], :] = corr_func[func2apply](data_proc[ifield][np.where(irows==i)[0]], axis=0)
                            
                    ikept[np.where(irows==i)[0][0]] = True
    
            
            elif proc[1] == 'repro':
                temp_ID = scan_ID - (xfactor[-3] * data_proc['iRepro'] + xfactor[-2] * data_proc['iRepet'] + xfactor[-1] * data_proc['iScan'])
            
                # Nécessaire pour éviter des problèmes d'arrondis qui 
                # engendrent deux nombres quasi identiques 
                # (e.g. 200.0001 et 200.00010000000001)
                if round_method == 'Banker':
                    temp_ID = np.round(temp_ID * 1 / np.min(xfactor)) * np.min(xfactor)  
                else:
                    temp_ID = RoundUp(temp_ID * 1 / np.min(xfactor)) * np.min(xfactor)
                    
                # Retrouve les lignes correspondant à toutes les repros d'un même 
                # produit
                unique_ID, irows = np.unique(temp_ID, return_inverse=True)
            
                # Initialise un vecteur de False pour savoir quelles sont les 
                # lignes qui seront conservées par la suite
                ikept = np.full(temp_ID.shape, False)
            
                # Pour chaque groupe de repro, procède à la moyenne des données
                for i in range(len(unique_ID)):
                    for ifield in ['mesure', 'T_DCT', 'T_DCTPCB', 'T_SMPL', \
                                   'T_SRC', 'VSRC', 'ISRC', 'DARK', \
                                       'VMEMS_SET', 'VMEMS_MES']:
                        if ifield in data_proc:
                            if len(data_proc[ifield].shape) == 1:
                                data_proc[ifield][np.where(irows==i)[0][0]] = corr_func[func2apply](data_proc[ifield][np.where(irows==i)[0]], axis=0)
                            else:
                                data_proc[ifield][np.where(irows==i)[0][0], :] = corr_func[func2apply](data_proc[ifield][np.where(irows==i)[0]], axis=0)
                            
                    ikept[np.where(irows==i)[0][0]] = True
    
            # On supprime les lignes qui ne sont plus nécessaires après moyenne
            for key, val in data_proc.items():
                if not key in shared_fields:
                    data_proc[key] = data_proc[key][ikept]
    
        
        # Concatène plusieurs structures de données en assurant que leurs 
        # champs soient identiques 
        elif proc[0] == 'concat':
            # 'processing_list' garde normalement une copie des paramètres de 
            # chaque traitement 
            # Pour éviter de conserver une copie complète de la structure 
            # complète concaténée (qui gaspillerait de la mémoire) on remplace
            # la structure par 'user_data'
            concat_mask = []
            for i in range(len(processing_list)):
                if processing_list[i][0] == 'concat':
                    concat_mask.append(i)
            processing_list[concat_mask[-1]] = ['concat', 'user_data']
            
            # La liste "procs" va contenir les n structures (sous la 
            # forme de dictionnaires) qui sont à concaténer ensemble
            procs = []
            
            # On ajoute à 'procs' la structure fournie par l'utilisateur
            procs.append(deepcopy(data_proc))
            
            # On ajoute à 'procs' les structures à concaténer avec la 
            # structure fournie par l'utilisateur
            for i in range(len(proc[1])):
                procs.append(proc[1][i])
               
            # On vérifie que toutes les dictionnaires (structures) ont les 
            # mêmes clés, on note quelles sont les clés en commun et celles
            # qui diffèrent (ici * permet de 'décompresser' les listes 
            # contenues dans 'procs_fields' etc)
            
            # Liste des champs de chaque structure
            procs_fields = [list(proc.keys()) for proc in procs]  
            
            # Liste de tous les champs rencontrés
            procs_fields_uniq = sorted(set().union(*procs_fields))  
            
            # Liste des champs manquants pour chaque structure
            missing_fields = [list(set(procs_fields_uniq) \
                                   - set(fields)) for fields in procs_fields]  
            
            # Liste de tous les champs manquants
            missing_fields_uniq = sorted(set().union(*missing_fields)) 
            
            # Complète les structures avec les champs qui manquent 
            for ifield in missing_fields_uniq:
                # Retrouve les structures chargées qui contiennent 
                # le champ manquant
                index = [i for i, proc in enumerate(procs) if ifield in proc]
                
                for ii in [i for i, fields in enumerate(missing_fields) if ifield in fields]:
                    # Boucle sur chaque structure qui ne contient pas le champ manquant en cours
                    if ifield in ['prop_name', 'pred_name', 'pred_model_id', \
                                  'info_name', 'subset_name']:
                        # Champs d'une seule et unique ligne (nom des 
                        # propriétés par exemple)
                        procs[ii][ifield] = procs[index[0]][ifield]
                    
                    elif ifield == 'info':
                        procs[ii][ifield] = np.tile('N/A', \
                                            (len(procs[ii]['prodname']), \
                                             procs[index[0]][ifield].shape[1]))
                    
                    elif ifield == 'subset':
                        procs[ii][ifield] = \
                            np.zeros((len(procs[ii]['prodname']), \
                                      procs[index[0]][ifield].shape[1]),\
                                      dtype=bool)
                    
                    else:
                        procs[ii][ifield] = np.nan * \
                            np.ones((len(procs[ii]['prodname']), \
                                     procs[index[0]][ifield].shape[1]))
            
            # Ajoute un champ iFile à chaque structure pour garder la 
            # distinction entre les différentes structures d'origine
            for i in range(len(proc)):
                procs[i]['iFile'] = procs[i]['iFile'] + i
            
            # On garde en mémoire les champs qu'on peut concaténer ensemble
            root_fields = list(procs[0].keys())
            root_fields = set(root_fields) - set(shared_fields)     
            
            # On concatène les structures (dictionnaires) contenus dans 'procs'
            # (qui ont maintenant exactement les mêmes clés)
            dict_concat = {}
            for key, val in procs[0].items():
                dict_concat[key] = [val]
            for struct in procs[1:]:
                for key, val in struct.items():
                    dict_concat[key].append(val)       
            procs = dict_concat
            
            # Vérifie que le contenu des champs 'shared_fields' est identique 
            # pour toutes les structures contenues dans 'procs'
            for ifield in shared_fields:
                # on ne teste pas 'processing_list' qui lui a le droit d'être 
                # différent 
                if ifield in procs and ifield != 'processing_list':
                    # Vérifie que le champ étudié est identique pour tous les
                    # éléments de de procs
                    identical_fields = \
                        bool(all(np.array_equal(procs[ifield][0], m) \
                        for m in procs[ifield]))
                    
                    # Si les champs ne sont pas indentiques, on fait en sorte 
                    # qu'ils le deviennent 
                    if not identical_fields:               
                        
                        if ifield == 'prop_name':
                            # Convertir les listes de chaînes de caractères en 
                            # une liste homogène
                            flat_list = [item for sublist in procs[ifield] \
                                         for item in sublist]
                            
                            # Utiliser np.unique sur la liste homogène
                            field_content_uniq = np.unique(flat_list)
                                        
                            # On rajoute les 'prop_name' manquant (si il y
                            # en a) et on ajoute à 'prop' un Nan associé aux
                            # indices des 'prop_name' ajoutés
                            for i in range(len(procs['prodname'])):
                                linked_field_tmp = \
                                    np.full((len(procs['prodname'][i]), \
                                             len(field_content_uniq)), np.nan)
                                
                                mask = np.isin(procs[ifield][i], \
                                               field_content_uniq)
                                icol = np.where(mask)[0]
                                
                                # Attribution des valeurs aux colonnes 
                                # correspondantes
                                for ii, col in enumerate(icol):
                                    linked_field_tmp[:, col] = \
                                            procs['prop'][i][:, ii]
                                
                                procs['prop'][i] = linked_field_tmp
                                procs[ifield][i] = field_content_uniq
                        
                        # !!! CAS A COMPLETER !!! (même idée que pour 
                        # ifield == 'prop_name' mais pas encore complété car
                        # la partie implémentant 'subset_name' n'existe pas 
                        # encore)   
                        elif ifield == 'subset_name':
                            raise ValueError("Warning ! Case ifield == 'subset_name' isn't coded yet")
                    
            # Concatène le struct array pour obtenir une seule et 
            #unique structure
            for ifield, arrays in procs.items():
                data_proc[ifield] = np.concatenate(arrays)
                
            # Traîte les champs dont la valeur est identique pour toutes les
            # acquisitions (on garde par défault la valeur de la première 
            # structure)
            for ifield in shared_fields:
                # vérifie que le champ existe bien (pas nécessairement vrai 
                # car "shared_fields" contient la liste de TOUS ces champs 
                # potentiellement présents)
                if ifield in procs: 
                    data_proc[ifield] = procs[ifield][0]
            
        
        # Permet d'identifier les subsets servant de source et de destination
        elif proc[0] == 'change_subset':
            # Liste de lignes à déplacer 
            irows = proc[1][0]
            
            # Récupère les subsets servant de source et de destination
            sub = re.findall(r'(.+)?->(.+)?', proc[1][1])
            source = np.array(sub[0][0])
            target = np.array(sub[0][1])
            
            # Crée ".subset" et ".subset_name" si nécessaire
            if 'subset_name' not in data_proc:
                data_proc['subset_name'] = np.array([])
                data_proc['subset'] = np.array([])
                
            source = np.array([]) if source == '' else np.array(source)
            target = np.array([]) if target == '' else np.array(target)
            
            # Ajoute les subsets manquants si nécessaire       
            # Subset à modifier
            sub2mod = [mat for mat in [source, target] if mat.size > 0]  
            
            # Subset à ajouter car manquant
            sub2add = np.logical_not(np.isin(sub2mod, data_proc['subset_name']))  
            
            if np.any(sub2add):
                subn = np.array(sub2mod)[sub2add]
                data_proc['subset_name'] = subn if data_proc['subset_name'].size == 0 else np.concatenate([data_proc['subset_name'], subn])
                subs = np.zeros((data_proc['prodname'].shape[0], np.sum(sub2add)))
                data_proc['subset'] = subs if data_proc['subset'].size == 0 else np.concatenate([data_proc['subset'], subs], axis=1)
   
            # Les lignes sélectionnées par irows sont mises à 0 pour la source 
            # et à 1 pour la destination
            if source:
                data_proc['subset'][irows, data_proc['subset_name'] == source] = 0
            if target:
                data_proc['subset'][irows, data_proc['subset_name'] == target] = 1
            
            processing_list[-1][1] = np.array(processing_list[-1][1], dtype=object)
            
        
        elif proc[0] == 'sort_fields':
            field_order = ['prodname','filename','sensor','time', 'iFile',\
                'iFolder','iAcqui','iScan','iRepro','iRepet', 'wave',\
                'mesure','iglitch', 'processing_list', 'subset_name',\
                'subset','info_name','info','prop_name','prop','pred_name',\
                'pred_model_id','pred', 'AVERAGECOUNT','GAIN','CORRTEMPH',\
                'REDGAIN','REDOFFSET','TREAT','T_DCT','T_DCTPCB','T_PROC',\
                'T_SMPL','T_SRC','VSRC','ISRC','DARK','VMEMS_SET','VMEMS_MES',\
                'MES_STDx10']
                
            sorted_keys = {}
            for key in field_order:
                if key in data_proc.keys():
                    sorted_keys[key] = data_proc[key]
            
            data_proc = sorted_keys
            
        else:
            raise ValueError("Warning ! The process", proc[0], "doesn't exist")
    
    # On met à jour 'processing_list'
    data_proc['processing_list'] = np.array(processing_list, dtype=object)
    
    return data_proc