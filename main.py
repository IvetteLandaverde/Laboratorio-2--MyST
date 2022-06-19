
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                               -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

#--------- LIBRERIAS Y DATOS --------- #
import data as dt
import numpy as np
import pandas as pd
import visualizations as vz
import json
import random
import functions as fn 


# -------- Tests modelo APT midprice: -------
# experimento 1:
fn.experimento_1_midprice(dt.ob_data)
#experimento 2: (por minuto)
fn.df_exp2_2
#visualizaciones experimento 2:
#vz.grafica_midprice(data=fn.df_exp2_2, price_type='Mid Price',colors =['orange', 'red'])


# -------- Tests modelo APT weighted midprice: -------
# experimento 1:
fn.experimento_1_w_midprice(dt.ob_data)
#experimento 2: (por minuto)
fn.df_exp2_2_w
#visualizaciones experimento 2:
#vz.grafica_midprice_w(data = fn.df_exp2_2_w, price_type = 'Mid Price', colors = ['orange', 'red'])

