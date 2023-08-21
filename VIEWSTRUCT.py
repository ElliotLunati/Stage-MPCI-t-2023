# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:50:46 2023

@author: lunati

VERSION 1.0
"""


# Classe qui permet de parcourir les structures de données plus aisément, pour
# tracer des graphiques plus particulièrement
# Par exemple on pourra écire data.raw.prod.mesure pour parcourir les 
# dictionnaires
class ViewStruct:
    def __init__(self, dic):
        for key, value in dic.items():
            if isinstance(value, dict): # vérifie type de l'objet spécifié 
                setattr(self, key, ViewStruct(value))
            else:
                setattr(self, key, value)
                
