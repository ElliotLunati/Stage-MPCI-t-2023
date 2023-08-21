# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:36:25 2023

@author: lunati

VERSION 1.0
"""

from datetime import datetime, timedelta


def now():
    """
    :return nb_days: renvoie le nombre de jours écoulés depuis l'an 0
    """
    now = datetime.now()
    y_0 = datetime(1, 1, 1)
    diff = now - y_0
    nb_sec = diff.total_seconds()
    nb_days = nb_sec / 86400               
    
    return nb_days


def datestr(date):
    """
    :param date: date (en jour depuis l'an 0) \n
    :return datestr: renvoie la date d'entrée dans le format année / mois ...
    """
    
    y_0 = datetime(1, 1, 1)
    delta = timedelta(days=date)
    datestr = y_0 + delta

    return datestr
    