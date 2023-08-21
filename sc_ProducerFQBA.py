# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 10:27:41 2023

@author: lunati

VERSION 1.01

--- Change log ---

V-1.01 : Il est désormais possible d'importer tous les fichiers configs 
         d'un répertoire d'une sonde FQBA
         
"""

from tkinter import Tk
from tkinter import filedialog as fd
import time
from ImportData_ProdFQBA import ImportData_ProdFQBA
from scipy import io
import re
import os
import sys
import VIEWSTRUCT
from LoadData import LoadData
from CheckRampe import check_rampe


""" INPUTS """

# Chemin depuis la racine contenant les données de production des sondes FQBA
start_path = 'C:/Users/Elliot Lunati/Desktop/stage MPCI été 2023/python SP3H/version github/PRODUCTION'

# Chemin complet vers config de la sonde MASTER
master_config_path = 'C:/Users/Elliot Lunati/Desktop/stage MPCI été 2023/python SP3H/version github/PRODUCTION/FAB-08-002/MASTER/FQBA-002-0195/config.mat'

# Création de la fenêtre tkinter
root = Tk()
root.withdraw()
root.wm_attributes('-topmost', 1)

# Méthode d'arrondi 
# Nouvelle norme : 'Banker'
# Norme Matlab 2013 : 'RoundUp'
round_method = 'RoundUp'

# Choix des tâches à accomplir   
chosen_tasks = [
                'ImportData', 
                #'LoadData',      # Doit être utilisé seul
               ]

if not chosen_tasks:
    sys.exit()

""" SELECTIONS DES DOSSIERS / FICHIERS .mat """

choose = False
folder_path = None
while not choose:
    if 'LoadData' in chosen_tasks:
        folder_path = fd.askopenfilename()
    else:
        folder_path = fd.askdirectory()
    
    # On vérifie si le répertoire sélectionné correspond à une Config
    # ie le chemin depuis la racine jusqu'au répertoire se termine par
    # 'Config_xx' (9 derniers caractères)
    config_folder = re.findall(r'Config', folder_path[-9:])
    
    # On vérifie si le répertoire sélectionné correspond à une sonde FQBA
    # ie le chemin depuis la racine jusqu'au répertoire se termine par
    # 'FQBA-xxx-xxxx' ou 'FQBA-xxx-xxx' (13 derniers caractères)
    fqba_folder = re.findall(r'FQBA-\d{3}-\d{3,4}', folder_path[-13:])
    
    # On vérifie si le fichier est un .mat
    mat_file = folder_path.endswith('.mat')
    
    if not (config_folder or fqba_folder or mat_file or folder_path == ''):
        print('You need to select a valid folder / file (either a FQBA / Config folder or a .mat file)')
    else:
        choose = True
    
    if folder_path == '':
        print('--------------------------------------------')
        print('Dropping folder / file selection')
        sys.exit()

""" EFFECTUE LES TACHE DEMANDEES """      

start = time.time()     
if 'ImportData' in chosen_tasks:   
    if config_folder or fqba_folder:
        Import_start = time.time()
        print('--------------------------------------------')
        print('Importing data ...  \n')
            
        imported_data = {}
        imported_config = {}
        
        # On vérifie que le répertoire sélectionné correspondant à celui
        # d'une Config
        if config_folder:
            num_config = 'Config_' + folder_path[-2] + folder_path[-1]
            folder_data = ImportData_ProdFQBA(folder_path, start_path, round_method) 
            imported_data[num_config] = folder_data[0]
            imported_config[num_config] = folder_data[1]
    
        # Sinon on vérifie que le répertoire sélectionné correspondant à celui
        # d'une sonde FQBA
        elif fqba_folder:
            configs = os.listdir(folder_path)     
            for config in configs:
                num_config = 'Config_' + config[-2] + config[-1]
                config_complete_path = folder_path + '/' + config
                folder_data = ImportData_ProdFQBA(config_complete_path, start_path, round_method) 
                imported_data[num_config] = folder_data[0]
                imported_config[num_config] = folder_data[1]                                 
    
        Import_end = time.time()
        print('--------------------------------------------')
        print('Data loaded in', Import_end - Import_start, 'sec \n')
    else:
        raise ValueError('You must either select a FQBA / Config folder to use ImportData')
        
if 'LoadData' in chosen_tasks:
    if mat_file:
        Load_start = time.time()
        print('Loading data ...')
        data = io.loadmat(folder_path, \
               squeeze_me=True, struct_as_record=False, chars_as_strings=True)
        data = VIEWSTRUCT.ViewStruct(data)
        Load_end = time.time()
        print('--------------------------------------------')
        print('Data loaded in', Load_end - Load_start, 'sec')
    else:
        raise ValueError('You must select a .mat file to use LoadData')
      
end = time.time()

print('####################################')
print('End of program')
print('Execution in', end - start, 'sec')
print('####################################')