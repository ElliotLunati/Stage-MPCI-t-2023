# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 11:09:19 2023

@author: lunati

VERSION 1.04

--- Change log ---

V-1.01 : Changement de la manière dont on récupère le bkg.
         Toutes les données dont nous avons pas de valeur associée valent par
         défault Nan.
V-1.02 : Ajout de la possibilité de choisir la méthode d'arrondi (par défault
         python utilise le "Banker's rounding" alors que dans la version de
         matlab 2013 c'est le "Round up" qui est utilisé). Ainsi par soucis
         de rétro-compatibilité on laisse ce choix (à noter que le "Round up"
         est 100x plus lent mais ne rajoute que 1/2 sec à l'exécution de
         l'importation des données).
V-1.03 : Création du fichier .py 'ImportData_Repertory' pour alléger le code
V-1.04 : Possibilité d'importer tous les scans (et pas la moyenne des acquis)

"""

import os
from scipy import io
import numpy as np
import re
import pandas as pd
from now import now
from copy import deepcopy
from process_data import process_data
from ImportData_Folder import ImportData_Folder
from process_config import process_config


############################## PREPROC ###############################
                
round_method = '' # !! Valeur assignée dans la fonction ImportData_ProdFQBA (et modifiable dans sc_ProducerFQBA)

# Chemin complet vers la BDD : Master BDD
Master_BDD_path = 'C:/Users/Elliot Lunati/Desktop/stage MPCI été 2023/python SP3H/version github/PRODUCTION/BDD Information+HCP+Propriétés.xlsb'

bkg_proc = [] # N'est pas utilisé pour l'instant
raw_resampling =  0 # N'est pas utilisé pour l'instant
raw_glitch_filter = 0 # Permet de filtrer les valeurs abhérentes
raw_glitch_threshold = 40 # Si vaut 'prod' on prend comme threshold celui spécifié dans le .csv
raw_average = 1 # Permet de moyenner les mesures

#######################################################################

############################### INPUT ################################

bkg_file = '' # Par défault on prend comme bkg la ligne REFERENCE d'un .cvs
bkg_data = 'prod' # Par défault si on force un fichier pour le spectre de référence ce sont les données produits qui servent de bkg, pas la ligne REFERENCE
tps_max_btw_repet = 1/144 # 10min (en nb jours) max entre deux repet d'un même produit
sensor_type = 'FQBx' # N'est pas utilisé pour l'instant
file2update = '' # N'est pas utilisé pour l'instant
read_every_nfiles = 1 # N'est pas utilisé pour l'instant

#######################################################################


###############################
###### FONCTIONS ANNEXES ######
###############################


def mat_file_exists(data_path):
    """
    :param data_path: répertoire dans lequel on vérifie 
    l'existance d'un fichier .mat \n
    
    :return bool: True si un fichier .mat existe, False sinon
    """
    
    files = os.listdir(data_path)
    for file in files:
        if file.endswith('.mat'):
            return True
    return False


def check_format(file_path):
    """
    :param file_path: dossier dans data_path (répertoire dans lequel 
    les données que l'on souhaite analyser se trouvent) dont on veut vérifier 
    si il contient des données. On vérifie si le début du nom du dossier est
    de la forme 23-05-26 (date) \n
    
    :return bool: True si config.mat existe, False sinon
    """
    
    pattern = r"\d{2}-\d{2}-\d{2}"
    eq = re.match(pattern, file_path[0:8])
    return bool(eq)


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


def swap_reference_element(lst):
    """
    :param lst: liste contenant le nom des répertoires dans lesquels on doit
    créer des fichiers .mat (qui contient le nom du répertoire référence)\n
    
    :return lst: renvoie une liste contenant les mêmes éléments, mais avec le
    nom du répertoire référence en tant que premier élément \n
    """
    
    reference_index = None

    # Recherche l'index de la chaîne contenant "REFERENCE"
    for i, element in enumerate(lst):
        if "REFERENCE" in element:
            reference_index = i
            break

    # Échange la position de la chaîne "REFERENCE" avec le premier élément
    if reference_index is not None and reference_index != 0:
        lst[0], lst[reference_index] = lst[reference_index], lst[0]

    return lst


def set_mat_data_name(folder_path, data_path, start_path):
    """
    :param folder_path: chemin vers un réportoire dans lequel on veut créer un 
    fichier .mat \n
    :param data_path: répertoire dans lequel les données que l'on souhaite
    analyser se trouvent (chemin jusqu'à un répertoire config en général) \n
    :start_path: chemin depuis la racine contenant les données de production 
    des sondes FQBx \n
    
    :return new_file_name: nom du fichier .mat qu'on va enregistrer
    """
    
    current_string = data_path.replace(start_path + '/', '')
    
    # Recherche du motif 'FQBx-xxx-xxxx'
    match_fqbx = re.findall(r'FQB\w-\d{3}-\d{4}', current_string)
    if match_fqbx:
        fqbx_code = match_fqbx[0]
    else:
        fqbx_code = ''

    # Recherche du motif 'Config_xx'
    match_config = re.findall(r'Config_\d{2}', current_string)
    if match_config:
        config_num = match_config[0][-2:]
    else:
        config_num = ''
    
    current_string = folder_path.replace(data_path + '/', '')
    
    # Recherche du motif correspondant à la date sous la forme xx-xx-xx
    date = current_string[:8]
    
    # Recherche du motif correspondant au nom du répertoire dans lequel on 
    # veut enregistrer un fichier .mat
    folder_name = current_string.replace(date, '')
    
    new_file_name = date + ' ' + fqbx_code + ' ' + 'C' + config_num + folder_name + '.mat'
    
    return new_file_name


def mat_struct(raw_prod, raw_bkg, preproc):
    """
    :param raw_prod: dictionnaire contenant les informations de la sous 
    structure 'prod' de la strucutre 'raw' d'un fichier .mat \n
    :param folder_bkg: dictionnaire contenant les informations de la sous 
    structure 'bkg' de la strucutre 'raw' d'un fichier .mat \n
    :param preproc: dictionnaire contenant les preprocs utilisés \n
        
    :return struct: renvoie le dictionnaire sous la bonne structure
    """
    
    struct = {'preproc': preproc,
              'raw': {'prod': raw_prod, 
                      'bkg': raw_bkg}
             }
    
    return struct


#################################
###### FONCTION PRINCIPALE ######
#################################


def ImportData_ProdFQBA(data_path, start_path, round_method):
    """
    :param data_path: répertoire dans lequel les données que l'on souhaite
    analyser se trouvent (chemin jusqu'à un répertoire config en général) \n
    :param start_path: chemin depuis la racine contenant les données de 
    production des sondes FQBA \n
    :param rounding_method: méthode d'arrondi à utiliser ('Banker' pour la norme
    actuelle et 'RoundUp' pour celle de Matlab 2013 \n
    
    :return mat_data: renvoie un dictionnaire contenant les données des 
    fichiers .mat dans le répertoire étudié \n
    :return config_data: renvoie les données du fichier config du répertoire 
    étudié \n
    """
    
    # On vérifie si le fichier config.mat existe déjà, si c'est le cas on 
    # récupère les données qu'il contient 
    if mat_file_exists(data_path):
        config = io.loadmat(data_path + '/config.mat', \
                squeeze_me=True, struct_as_record=False, chars_as_strings=True)
        
        print('--------------------------------------------')
        print('Config file imported from', data_path + '/config.mat', '\n')
        
        # On transforme la config et ses infos en dictionnaire
        config = process_config(config, [['config2dict', []]])
    
    # Sinon on doit le créer 
    # On garde en mémoire la date de création du nouveau fichier config.mat       
    else:
        config = {'TimeCreation': now()}
        print('--------------------------------------------')
        print('No config file found in', data_path)
        print('A config file has been created \n')
        
    # On cherche les sous répertoires où les données 
    # sous format .csv se trouvent

    folders = os.listdir(data_path)
    valid_data = []
    for folder in folders:
        if check_format(folder):
            valid_data.append(data_path + '/' + folder)
    
    # On identifie les répertoires où on doit créer les fichiers .mat (si il
    # en existe)
    data_create_mat = []
    
    # On vérifie si il existe ou non un fichier .mat dans le répertoire 
    # référence pour pouvoir le créer un priorité si c'est pas le cas
    mat_in_ref = False
    
    for folder in valid_data:
        if not mat_file_exists(folder):
            data_create_mat.append(folder)
        elif not mat_in_ref and \
            'REFERENCE' in folder.replace(data_path + '/', ''):
            mat_in_ref = True
                
    # On importe les données de la Master BDD (environ 3sec) et on parcourt 
    # les fichiers dans les répertoires contenus dans 'data_create_mat' 
    # seulement 'data_create_mat' si n'est pas vide, ie qu'on doit créer des
    # fichiers .mat dans ces répertoires et donc qu'on doit avoir accès à la 
    # Master BDD
    if len(data_create_mat) != 0:     
        print('--------------------------------------------')
        print('Loading data of the MasterBDD from', Master_BDD_path)
        data_Master_BDD = pd.read_excel(Master_BDD_path, engine='pyxlsb', index_col=0)
    
        # On change la configuration de la BDD pour pouvoir plus facilement 
        # accèder aux données
        Master_BDD_T = data_Master_BDD.T
        Master_BDD = pd.DataFrame([line for line in Master_BDD_T.values[1:]],\
            columns=Master_BDD_T.values[0], index=data_Master_BDD.columns[1:])
        print('MasterBDD successively imported \n')
        
        # Initialisation des preprocs à appliquer
        preproc = {'prod': [Master_BDD_path, Master_BDD],
                   'bkg_proc': bkg_proc,
                   'raw_resampling': raw_resampling,
                   'raw_glitch_filter': raw_glitch_filter,
                   'raw_glitch_threshold': raw_glitch_threshold,
                   'raw_average': raw_average,      
                   'round_method': round_method
                   }
            
        # Initialisation des inputs
        inputs = {'bkg_file': bkg_file,
                  'bkg_data': bkg_data,
                  'tps_max_btw_repet': tps_max_btw_repet,
                  'sensor_type': sensor_type,
                  'file2update': file2update,
                  'read_every_nfiles': read_every_nfiles     
                  }
        
        # On fait en sorte d'importer les données du répertoire référence pour 
        # pouvoir compléter la structure bkg
        # Si un fichier de référence est donné au préalable, on fait en sorte
        # de ne pas l'écraser plus tard en fixant 'bkg_found' à True
        bkg_found = False
        if inputs['bkg_file']:
            bkg_found = True
       
        # Si un fichier .mat existe dans le répertoire de référence mais que
        # la config ne pointe pas vers le fichier de référence, on fait en
        # sorte que ça soit le cas
        if mat_in_ref:    
            if not 'Reference' in config or not 'file_csv_raw' in config['Reference']:
                config = process_config(config, [['update_links', data_path]])    
                
        # Si le fichier .mat du répertoire référence n'existe pas, on le crée 
        # en priorité (i.e. on place le nom du fichier référence )
        else:
            data_create_mat = swap_reference_element(data_create_mat)
            
        # Si la config pointe vers un fichier référence et qu'aucune référence
        # n'a été forcée par l'utilisateur, alors on utilise comme référence
        # celle du fichier config
        if (not bkg_found) and ('Reference' in config):
            if 'file_csv_raw' in config['Reference']:
                ref_file = config['Reference']['file_csv_raw']
                inputs['bkg_file'] = ref_file
                bkg_found = True
                  
        # On sauve les données contenues dans les sous répertoires valides dans 
        # le format .mat (qu'on crée dans le même répertoire)
        # Si un .mat existe déjà dans le dossier, on ne fait rien
        
        for folder in data_create_mat: 
            print('--------------------------------------------')
            print('Importing .csv files data from', folder, '\n')
            
            # Importation des données du dossier
            folder_info = ImportData_Folder(folder, preproc, inputs)
            folder_data = folder_info[0]
            folder_bkg = folder_info[1]
            
            # Bien que les fichiers soient normalement lus dans l'ordre 
            # chronologique, ce premier tri se fait sur la base de la date de
            # modif windows qui peut ne pas être fiable
            # On fait donc une seconde passe pour s'assurer que les données 
            # sont bien dans l'ordre chronologique
            ichrono = np.argsort(folder_data['time'])
            folder_data = process_data(folder_data, [['apply_query', ichrono]])
            folder_bkg = process_data(folder_bkg, [['apply_query', ichrono]])
            
            # On fait en sorte que 'iAcqui' soit de nouveau dans l'ordre
            jchrono = np.argsort(ichrono)
            folder_data['iAcqui'] = folder_data['iAcqui'][jchrono]
            folder_bkg['iAcqui'] = folder_bkg['iAcqui'][jchrono]
            
            # On ordonne les clés du dictionnaire 'folder_data' et 'folder_bkg'
            folder_data = process_data(folder_data, [['sort_fields', []]])
            folder_bkg = process_data(folder_bkg, [['sort_fields', []]])
            
            # On enlève les valeurs de la MasterBDD des preprocs enregsitrés
            # dans le .mat pour économiser de la mémoire
            preproc_saved = deepcopy(preproc)
            preproc_saved['prod'][1] = 'MasterBDD data'
            
            # On crée une structure dans laquelle on place les données 
            # importées
            folder_data = mat_struct(folder_data, folder_bkg, preproc_saved)
               
            # On accède au répertoire associé aux données extraites et on 
            # crée un fichier .mat contenant 'folder_data'
            save_directory = folder + '/'
            os.makedirs(save_directory, exist_ok=True)
            
            # On construit à partir de 'folder' le nom du fichier .mat qu'on 
            # va enregistrer
            save_path = os.path.join(save_directory, \
                                set_mat_data_name(folder, data_path, start_path))
            io.savemat(save_path, folder_data)
          
            print('--------------------------------------------')
            print('Imported data from', folder, 'saved in the .mat file', save_path, '\n')
          
            config = process_config(config, [['update_links', folder]])
            
            # Si on n'avait pas détecté de fichier .mat dans la référence au 
            # préalable, et que la config ne pointait vers aucun fichier de 
            # référence, alors le premier répertoire qu'on a traité est celui
            # de la référence et on a donc maintenant accès à un fichier de
            # background
            if not bkg_found:
                bkg_found = True
                inputs['bkg_file'] = config['Reference']['file_csv_raw']
        
    # On charge les fichiers .mat des répertoires valides
    mat_data = {}
    for folder in valid_data:
        files = os.listdir(folder)
        for file in files:
            if file.endswith('.mat'):
                mat_file_path = folder + '/' + file
                print('--------------------------------------------')
                print('Loading data from the .mat file :', mat_file_path)
                
                loaded_data = io.loadmat(mat_file_path, \
                squeeze_me=True, struct_as_record=False, chars_as_strings=True)
       
                # On ajoute le dictionnaire contenant une partie des données 
                # importées dans le dictionnaire des données importées
                mat_data[file] = loaded_data 
                print('Data loaded successively \n')
        
    # 0n met à jour les liens de la config
    config = process_config(config, [['update_links', data_path]])
    
    # Sauvegarde de config dans un fichier .mat
    process_config(config, [['save_config', data_path]])
     
    return mat_data, config


