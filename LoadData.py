# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:43:07 2023

@author: lunati

VERSION 1.0
"""

from scipy import io
import VIEWSTRUCT
import sys
from tkinter import filedialog as fd

def LoadData():
    """
    :return data: charge les données d'un fichier .mat dans la classe ViewStruct
    """
    
    choose = False
    file_path = None
    while not choose:
        file_path = fd.askopenfilename()
        
        # On vérifie si le fichier est un .mat
        mat_file = file_path.endswith('.mat')
        
        if not (mat_file or file_path == ''):
            print('You need to select a .mat file')
        else:
            choose = True
        
    if file_path == '':
        print('--------------------------------------------')
        print('Dropping file selection')
        sys.exit()
        
    data = io.loadmat(file_path, \
           squeeze_me=True, struct_as_record=False, chars_as_strings=True)
    data = VIEWSTRUCT.ViewStruct(data)
    
    return data