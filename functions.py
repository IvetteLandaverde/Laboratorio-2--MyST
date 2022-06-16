
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                               -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import data as dt
#import visualizations as vz
import numpy as np
import pandas as pd
import json

# read input data:
data_ob=dt.ob_data

#cantidad libros de ordenes que hay en total
n_libros = len(list(data_ob.keys()))
print(f"La cantidad total de libros de ordenes es: {n_libros}")

# -calcular el midprice

ob_ts=list(data_ob.keys())
l_ts= [pd.to_datetime(i_ts) for i_ts in ob_ts]
mid_price = [(data_ob[ob_ts[i]]['ask'][0]+data_ob[ob_ts[i]]['bid'][0])*0.5 for i in range(0,len(ob_ts))]

total = len(mid_price)-1

#NOTA: para contabilizar los e1 de acuerdo a estar formulacion : P_t = E[P_t +1]
e1 = [midprice[i] == midprice[i+1] for i in range(len(mid_price)-1)]
e2 = len(mid_price) -1 - sum(e1)

#guardar resultados de conteo y de proporcion en un diccionario
metricas = {'e1':{'cantidad': e1, 'proporcion': np.round(sum(e1)/total,2)},'e2':{'cantidad': e2, 'proporcion': np.round(e2/total,2)},'total': len(mid_price)-1}

#imprimir resultados
exp_1['e1']['proporcion']
exp_1s['e2']['proporcion']

# -Repetir lo anterior para otros (Experimentos con datos de cada minuto)
# Experimentos: 00:06:00 - 00:07:00 ... 00:05:00 - 00:06:00

# Hacer un dataframe para guardar los resultados finales 
#nota: el df hacerlo ya que haya datos
exp_2 = pd.DataFrame({'intervalo':list(np.arange(0,60)),'total':[],'e1_conteo':[],'e1_proporcion':[],'e2_conteo':[],'e2_proporcion':[]},index = list(np.arange(0,60)))

minutes = list(np.arange(0,60))

#para validar
#list(set([i_ts.minute for i_ts in l_ts]))

l_ts[0].minute == minutes[0] # si no pasa es q el timestamp no fue en el minuto 0

##

#Pendientes:
#1: hacer el ejemplo  sig pero en ciclo
#2: todo lo anterior tmb para weighted midprice
#3: Hacer una gráfica de analisis simple, entre exploratorio y descriptivo