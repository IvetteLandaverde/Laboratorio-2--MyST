
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project:High-Frequency Models a) Asset Pricing Theory b) Roll Model                                                         -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: IvetteLandaverde                                                                     -- #
# -- license: GNU General Public License v3.0                                            -- #
# -- repository: https://github.com/IvetteLandaverde/Laboratorio-2--MyST                                                               -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import data as dt
import visualizations as vz
import numpy as np
import pandas as pd
import json
import time
import random

# --------- FUNCIÓN SAMPLE FROM DICT ------- #
def sample_from_dict(d, sample=1):
    keys = random.sample(list(d), sample)
    values = [d[k] for k in keys]
    return dict(zip(keys, values))
#sample_from_dict(dt.ob_data)
# --------- FUNCIÓN SAMPLE FROM DICT ------- #




# --------- ASSET PRICING MODEL ------- #

# -- read input data:
data_ob=dt.ob_data


#-- cantidad ibros de ordenes que hay en total:
n_libros = len(list(data_ob.keys()))
print(f"La cantidad total de libros de ordenes es: {n_libros}")

# -- calcular el midprice:
ob_ts=list(data_ob.keys())
l_ts= [pd.to_datetime(i_ts) for i_ts in ob_ts]
mid_price = [(data_ob[ob_ts[i]]['ask'][0]+data_ob[ob_ts[i]]['bid'][0])*0.5 for i in range(0,len(ob_ts))]
# guardar el midprice para contabilizarlo con P_t = E[P_t +1]
# guardar info en un diccionario
total = len(mid_price)-1
e1 = [mid_price[i] == mid_price[i+1] for i in range(len(mid_price)-1)]
e2 = len(mid_price) -1 - sum(e1)
exp_1 = {'e1':{'cantidad': e1, 'proporcion': np.round(sum(e1)/total,2)},'e2':{'cantidad': e2, 'proporcion': np.round(e2/total,2)},'total': len(mid_price)-1}
# comprobar que tiene un comportamiento martingala:
# imprimir los resultados:
exp_1['e1']['proporcion']
exp_1['e2']['proporcion']


# -- Repetir lo anterior para otros experimentos con datos de cada minuto 
# experimentos   00:06:00 - 00:06:59 ... 00:05:00 - 00:05:59
# hacer un dict para guardar resultados finales:
exp_2 = {'intervalo':list(np.arange(0,60)),'total':[],'e1_conteo':[],'e1_proporcion':[],'e2_conteo':[],'e2_proporcion':[]}
# nota: hacer los calculos ahi mismo en el dataframe e irlo llenando con la info por minuto
# nota 2: se puede hacer dataframe hasta que ya tenga datos el diccionario 
exp_2 = pd.DataFrame({'intervalo':list(np.arange(0,60)),'total':[],'e1_conteo':[],'e1_proporcion':[],'e2_conteo':[],'e2_proporcion':[]},index = list(np.arange(0,60)))

# --------- ASSET PRICING MODEL ------- #




# ideas minutos
minutes = list(np.arange(0,60))# esto es lo que queremos valuar, del 1 al 59
[i_ts.minute for i_ts in l_ts]
#hago una lista para ver si ya todos sosn valores unicos, sino son arreglarlo
#list(set([i_ts.minute for i_ts in l_ts]))
# para saber si si son se igualan ambas:
#list(set([i_ts.minute for i_ts in l_ts])) == list([i_ts.minute for i_ts in l_ts])

#jwhek



#Pendientes:
#1: hacerlo por cada minuto
#2: todo lo anterior tmb para weighted midprice
#3: Hacer una gráfica de analisis simple, entre exploratorio y descriptivo
#4: Hacer el roll model 


# --------- ROLL MODEL ------- #

# --------- ROLL MODEL ------- #