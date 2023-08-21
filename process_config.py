# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 09:46:12 2023

@author: lunati

VERSION 1.0
"""

import os
import re
from now import now
from scipy import io
from copy import deepcopy
import numpy as np


###############################
###### FONCTIONS ANNEXES ######
###############################


def find_mat_files(directory):
    """
    :param directory: chemin vers le répertoire dans lequel on va trouver tous 
    les fichiers .mat (sous répertoires inclus) \n
    :return mat_files: renvoie tous les chemins vers les fichiers .mat trouvés 
    """
    mat_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mat'):
                mat_files.append(os.path.join(root, file))
    
    return mat_files


#################################
###### FONCTION PRINCIPALE ######
#################################


def process_config(config, proc_list):
    """
    :param config: dictionnaire contenant les données du fichier config 
    !!! APPLIQUER config2dict AVANT !!! \n
    :param proc_list: traîtement des données qu'on veut appliquer, en 
    fonction de la requête il peut être de la forme : \n
    - add_info : process_config(config, ['add_info', [field, subfield, field_val]]) 
      où par exemple field = 'Reference', subfield = 'data' et field_val la 
      valeur qu'on veut enregistrer \n
    - save_config : process_config(config, [['save_config', 'config_fpath]']) où  
      config_fpath est le chemin depuis la racine où on veut enregistrer la config \n
    - update_links : process_config(config, ['update_links', 'source_path']) où 
      source_path est le chemin depuis la racine vers le répertoire à partir duquel 
      on va mettre à jour les liens vers les différents fichiers .mat \n
    - config2dict : process_config(config, [['config2dict', []]]) \n
    
    :return data_proc: renvoie les données sous la forme d'un dictionnaire où
    les traîtements demandés ont été appliqués
    """
    
    proc_config = deepcopy(config)
    
    for proc in proc_list:          
       
        # Ajoute l'information souhaitée dans le fichier config (si celle-ci 
        # existe déjà elle sera écrasée par les nouvelles données)
        if proc[0] == 'add_info':
            field = proc[1][0]
            subfield = proc[1][1]
            field_val = proc[1][2]
            
            if field in proc_config:
                if subfield in proc_config[field]:
                    print('-----------------------------------------------------------------------')
                    print(proc_config[field][subfield], 'in', field ,'.', subfield, 'replaced by', field_val, '\n')
                    proc_config[field][subfield] = field_val
                elif subfield:
                    print('-----------------------------------------------------------------------')
                    print(field_val, 'added in', field ,'.', subfield, '\n')
                    proc_config[field][subfield] = field_val
                else:
                    print('-----------------------------------------------------------------------')
                    print(field_val, 'added in', field, '\n')
                    proc_config[field] = field_val
                
            else:
                if subfield:
                    print('-----------------------------------------------------------------------')
                    print(field_val, 'added in', field ,'.', subfield, '\n')
                    proc_config[field] = {subfield: field_val}
                else:
                    print('-----------------------------------------------------------------------')
                    print(field_val, 'added in', field, '\n')
                    proc_config[field] = field_val
            
            proc_config['TimeLastModif'] = now() 
        
        
        # Permet de sauver le fichier config en tant que .mat dans le répertoire
        # indiqué
        elif proc[0] == 'save_config':
            config_fpath = proc[1]
            os.makedirs(config_fpath, exist_ok=True)
            save_path = os.path.join(config_fpath, 'config.mat')
            io.savemat(save_path, proc_config)
            print('-----------------------------------------------------------------------')
            print('Config file saved in', config_fpath, '\n')
        
        
        # Cherche les fichiers .mat dans le répertoire indiqué et ses sous
        # répertoires directs et met à jour le fichier config si possible
        elif proc[0] == 'update_links':
            source_path = proc[1]
            source_path_depth = source_path.count('/')
            
            # Trouve tous les fichiers .mat contenus dans 'source_path'
            matlab_files = find_mat_files(source_path)
            
            # On ne garde que ceux contenus dans le répertoire et les 
            # sous-répertoires directs 
            matlab_fpath = [file for file in matlab_files if file.count('\\') <= source_path_depth + 2]
            
            # On récupère le nom des fichiers .mat (ce qui se trouve avant le 
            # dernier \)
            matlab_fname = [file.rpartition("\\")[-1] for file in matlab_fpath]
            
            """ FICHIERS DE DONNEES """
            
            # On garde les fichiers associés à des données (ie dont le nom 
            # commence par une date du type xx-xx-xx)
            pattern = r"^\d{2}-\d{2}-\d{2}"
            data_fpath = [matlab_fpath[i] for i in range(len(matlab_fname)) if re.match(pattern, matlab_fname[i])]
            data_fname = [string for string in matlab_fname if re.match(pattern, string)]
            
            # On récupère le nom du type de données 
            pattern = r'C\d{2}\s(\w+)'
            data_type = [re.findall(pattern, string)[0] for string in data_fname]
            
            data_type_list = ['SHIFTLO_DARK', 'BDD_TRANSFER',\
                              'REFERENCE', 'VALID', 'TCHARD', 'BDD_CALIB', 'TEST']
                              
            cfg_section = [['ShiftLO_MEMS', 'Dark'],\
                           ['SpcTransfer', 'Prop_predict'],\
                           ['CenterReduce', 'Reference'], ['Valid'], ['TcHard'],\
                           ['Prop_predict'], ['Test']] 
                               
            for i in range(len(data_fname)):
                try:
                    index = data_type_list.index(data_type[i])
                except:
                    raise ValueError(data_fname[i], 'isn\'t a valid data type. Please check the name of the data .mat files')
                      
                for j in range(len(cfg_section[index])):
                
                    # Pour Valid on peut avoir plusieurs fichiers dans la config
                    if cfg_section[index][j] == 'Valid':
                        if 'Valid' not in proc_config:
                            if not 'Valid' in proc_config:
                                proc_config['Valid'] = {}
                            proc_config['Valid']['data'] = np.array(data_fpath[i])
                        
                        # On ajoute l'élément à data ssi il ne s'y trouve pas déjà
                        elif not np.isin(data_fpath[i], proc_config['Valid']['data']):
                            proc_config['Valid']['data'] = np.append(proc_config['Valid']['data'], data_fpath[i])
                    
                    # Pour Test on peut avoir plusieurs fichiers dans la config
                    elif cfg_section[index][j] == 'Test':
                        if 'Test' not in proc_config:
                            if not 'Test' in proc_config:
                                proc_config['Test'] = {}
                            proc_config['Test']['data'] = np.array(data_fpath[i])
                        
                        # On ajoute l'élément à data ssi il ne s'y trouve pas déjà
                        elif not np.isin(data_fpath[i], proc_config['Test']['data']):
                            proc_config['Test']['data'] = np.append(proc_config['Test']['data'], data_fpath[i])       
                             
                    else:
                        # Sinon il ne peut y avoir qu'un fichier
                        if not cfg_section[index][j] in proc_config:
                            proc_config[cfg_section[index][j]] = {}
                        proc_config[cfg_section[index][j]]['data'] = data_fpath[i]
                    
                    print('-----------------------------------------------------------------------')
                    print('Path to data file', data_fpath[i], 'was added to', cfg_section[index][j], '.data \n')
                        
                # Dans le cas du fichier qui contient les spectres de REFERENCE
                # il faut aussi créer un lien vers le .csv qui servira pour le
                # passage en absorbance
                if data_type[i] == 'REFERENCE':
                    ref_data = io.loadmat(data_fpath[i], \
                    squeeze_me=True, struct_as_record=False, chars_as_strings=True)
                    
                    # On prend le dernier fichier mesuré lors de la RT à 25°C 
                    # (palier stabilisé) comme acqui de référence
                    ref_file = ref_data['raw'].prod.filename[-1]
                    proc_config['Reference']['file_csv_raw'] = ref_file
                    
                    print('-----------------------------------------------------------------------')
                    print('Path to data file', ref_file, 'was added to Reference.file_csv_raw \n')
                
                                
            """ FICHIERS DE MODELES """
            # A complétér
            
            """ FICHIERS LIES A LA REFERENCE UTILISEE POUR PASSAGE EN ABSORBANCE """
            
            ref_fname = [file for file in matlab_fname if '_Reference' in file]
            ref_fpath = [matlab_fpath[i] for i in range(len(matlab_fname)) if '_Reference' in matlab_fname[i]]
            if ref_fname:
                if not 'Reference' in proc_config:
                    proc_config['Reference'] = {}
                
                proc_config['Reference']['file_matlab'] = np.array(ref_fpath)
                print('-----------------------------------------------------------------------')
                print('Path to data file', ref_fpath[0], 'was added to Reference.file_matlab \n')
            
                proc_config['Reference']['csv_sensor'] = np.array(ref_fpath[0].replace('.mat', '.csv'))
                print('-----------------------------------------------------------------------')
                print('Path to data file', ref_fpath[0].replace('.mat', '.csv'), 'was added to Reference.csv_sensor \n')
                
                ref_data = io.loadmat(ref_fpath[0], \
                squeeze_me=True, struct_as_record=False, chars_as_strings=True)
                
                ref_file = ref_data['data'].filename
                proc_config['Reference']['file_csv_raw'] = np.array(ref_file)
                print('Path to data file', ref_file, 'was added to Reference.file_csv_raw \n')
                
            proc_config['TimeLastModif'] = now()
        
        # Transforme un config importée et ses infos en dictionnaires
        elif proc[0] == 'config2dict':
            config_dict = {}
            for info in proc_config:
                try:
                    config_dict[info] = {}
                    for fieldname in proc_config[info]._fieldnames:
                        config_dict[info][fieldname] = getattr(proc_config[info], fieldname)
                except:
                    config_dict[info] = proc_config[info]
            
            proc_config = config_dict
                
    return proc_config