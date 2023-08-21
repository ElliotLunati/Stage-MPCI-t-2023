# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:02:51 2023

@author: lunati

VERSION 1.0
"""

import decimal
import numpy as np


def RoundUp(array):
    """
    :param array: matrice numpy dont on veut arrondir les éléments en utilisant
    la méthode 'RoundUp' (celle de Matlab 2013) \n
    
    :return: renvoie la matrice arrondie avec la méthode 'RoundUp' (celle de Matlab 2013)
    """
    
    decimal.getcontext().rounding = decimal.ROUND_HALF_UP
    vectorized_to_integral_value = np.vectorize(lambda x: int(decimal.Decimal(str(x)).to_integral_value()))
    rounded_array = vectorized_to_integral_value(array)
    
    return rounded_array