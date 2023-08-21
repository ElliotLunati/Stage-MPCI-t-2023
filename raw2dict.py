# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:38:37 2023

@author: lunati

VERSION 1.0
"""


def raw2dict(data):
    """
    :param data: dictionnaire contenant les données d'un fichier .mat\n
        
    :return raw_prod_dict: dictionnaire contenant les données d'un fichier .mat 
    
    """
    
    raw_dict = {}
    for struct in data:
        try:
            raw_dict[struct] = {}
            for ifieldname in data[struct]._fieldnames:  
                attr = getattr(data[struct], ifieldname)
                try:
                    raw_dict[struct][ifieldname] = {}
                    for jfieldname in attr._fieldnames:
                       raw_dict[struct][ifieldname][jfieldname] = getattr(attr, jfieldname)
                    
                except:
                    raw_dict[struct][ifieldname] = attr
        except:
            del raw_dict[struct]
        
    return raw_dict