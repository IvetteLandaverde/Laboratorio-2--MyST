
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                               -- #
# -- repository: https://github.com/IvetteLandaverde/Laboratorio-2--MyST                                                               -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

#--------- LIBRERIAS Y DATOS --------- #
import data as dt
import numpy as np
import pandas as pd
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

#
#
#
# importar librerias y datos: 
import data as dt
import visualizations as vz
import numpy as np
import pandas as pd
import json
import time
import random
import plotly.graph_objects as go

# --------- FUNCIÓN SAMPLE FROM DICT ------- #
def sample_from_dict(d, sample=1):
    keys = random.sample(list(d), sample)
    values = [d[k] for k in keys]
    return dict(zip(keys, values))
#sample_from_dict(dt.ob_data)

# --------- ASSET PRICING MODEL ------- #
#######
#######
# --------- ASSET PRICING MODEL MIDPRICE: ------- #
######
#######
# -- read input data:
data_ob=dt.ob_data
ob_ts=list(data_ob.keys())#
l_ts= [pd.to_datetime(i_ts) for i_ts in ob_ts]#

# ------ hacer el experimento 1 en función  -----
def experimento_1_midprice(data_ob):
    n_libros = len(list(data_ob.keys()))
    #print(f"La cantidad total de libros de ordenes es: {n_libros}")
    # -- calcular el midprice:
    ob_ts=list(data_ob.keys())
    l_ts= [pd.to_datetime(i_ts) for i_ts in ob_ts]
    mid_price = [(data_ob[ob_ts[i]]['ask'][0]+data_ob[ob_ts[i]]['bid'][0])*0.5 for i in range(0,len(ob_ts))]
    # guardar el midprice para contabilizarlo con P_t = E[P_t +1]
    # guardar info en un diccionario
    total = len(mid_price)-1
    e1 = [mid_price[i] == mid_price[i+1] for i in range(len(mid_price)-1)]
    e2 = len(mid_price) -1 - sum(e1)
    exp_1 = {'e1':{'cantidad': sum(e1), 'proporcion': np.round(sum(e1)/total,2)},'e2':{'cantidad': e2, 'proporcion': np.round(e2/total,2)},'total': len(mid_price)-1}
    df_exp_1 = pd.DataFrame({'e1':{'cantidad': sum(e1), 'proporcion': np.round(sum(e1)/total,2)},'e2':{'cantidad': e2, 'proporcion': np.round(e2/total,2)},'total': len(mid_price)-1})
    # comprobar que tiene un comportamiento martingala:
    # imprimir los resultados:
    exp_1['e1']['proporcion']
    exp_1['e2']['proporcion']
    return df_exp_1
# ------ hacer el experimento 2 en función  -----
#primero crear un dataframe con el midprice:
df2 = pd.DataFrame()
df2['Timestamp'] = None
df2 = df2.assign(Midprices=None)
times = (l_ts)
mids = [(data_ob[ob_ts[i]]['ask'][0]+data_ob[ob_ts[i]]['bid'][0])*0.5 for i in range(0,len(ob_ts))]
df2['Timestamp'] = times
df2['Midprices'] = mids
df2 = df2.set_index('Timestamp')
#...
dic = {}
for index, row in df2.iterrows():
    key = str(index.hour) + ":" + str(index.minute)
    value = row["Midprices"]
    try:
        dic[key].append(value)
    except KeyError:
        dic[key] = [value]
# ahora si hacer las iteraciones
metricasMin = {}
for i in list(dic): 
    e1 = sum([dic[i][i_t] == dic[i][i_t+1] for i_t in range(len(dic[i])-1)])
    e2 = sum([dic[i][i_t] != dic[i][i_t+1] for i_t in range(len(dic[i])-1)])
    metricasMin[i] = (
        {"e1" :{"cantidad" :  e1, "proporcion" : e1/(len(dic[i])-1)}, 
         "e2" :{"cantidad" :  e2, "proporcion" : e2/(len(dic[i])-1)},
         "total" : len(dic[i])-1
         }
        )
tot = []
for i in list(metricasMin):
    tot.append(metricasMin[i]["total"])
np.array(tot).sum()
# Proporción de frecuencia de martingalas
mgE1 = []
for i in list(metricasMin):
    mgE1.append(metricasMin[i]["e1"]["cantidad"])
np.array(mgE1).sum()/np.array(tot).sum()
# Promedio de proporción de martingalas
propE1 = []
for i in list(metricasMin):
    propE1.append(metricasMin[i]["e1"]["proporcion"])
np.array(propE1).mean()
#datos para llenar df fnal (experimento 2):
df_exp2 = pd.DataFrame(metricasMin).T
df_e1_exp2=pd.DataFrame(df_exp2["e1"][i] for i in range(0,60))
df_e1_exp2_conteo = pd.DataFrame(df_e1_exp2["cantidad"])
df_e2_exp2=pd.DataFrame(df_exp2["e2"][i] for i in range(0,60))
df_e2_exp2_conteo = pd.DataFrame(df_e2_exp2["cantidad"])
df_e1_exp2_proporcion = pd.DataFrame(df_e1_exp2["proporcion"])
df_e2_exp2_proporcion = pd.DataFrame(1-df_e1_exp2["proporcion"])
df_exp2_total = pd.DataFrame(df_e1_exp2_conteo["cantidad"]+df_e2_exp2_conteo["cantidad"])
# llenar el df del experimento 2:
df_exp2_2 = pd.DataFrame()
df_exp2_2 = df_exp2_2.assign(e1=None)
df_exp2_2 = df_exp2_2.assign(e2=None)
df_exp2_2 = df_exp2_2.assign(total=None)
df_exp2_2 = df_exp2_2.assign(proporcion1=None)
df_exp2_2 = df_exp2_2.assign(proporcion2=None)
times = (l_ts)
valor_e1 = df_e1_exp2_conteo 
valor_e2 = df_e2_exp2_conteo 
valor_total = df_exp2_total
valor_proporcion1 = df_e1_exp2_proporcion 
valor_proporcion2 = df_e2_exp2_proporcion 
df_exp2_2['e1'] = valor_e1
df_exp2_2['e2'] = valor_e2
df_exp2_2['total'] = valor_e1+valor_e2
df_exp2_2['proporcion1'] = valor_proporcion1
df_exp2_2['proporcion2'] = valor_proporcion2




#######
#######
# --------- ASSET PRICING MODEL WEIGHTED MIDPRICE: ------- #
######
#######
# ------ hacer el experimento 1 en función  -----
def experimento_1_w_midprice(data_ob):
    n_libros = len(list(data_ob.keys()))
    #print(f"La cantidad total de libros de ordenes es: {n_libros}")
    # -- calcular el midprice:
    ob_ts=list(data_ob.keys())
    l_ts= [pd.to_datetime(i_ts) for i_ts in ob_ts]
    #
    ob_ts = list(data_ob.keys())
    ob_imb = lambda v, d: np.sum(v.iloc[:d, 0]) / np.sum([v.iloc[:d, 0], v.iloc[:d, 1]])
    mid = [(data_ob[ob_ts[i]]['ask'][0] + data_ob[ob_ts[i]]['bid'][0])* 0.5 for i in range(0, len(ob_ts))]
    imbalance = [ob_imb(data_ob[i_ts][['bid_size', 'ask_size']], len(data_ob[i_ts])) for i_ts in ob_ts]
    w_mid_price = list(np.array(mid) * np.array(imbalance))
    #
    # guardar el midprice para contabilizarlo con P_t = E[P_t +1]
    # guardar info en un diccionario
    total_w = len(w_mid_price)-1
    e1_w = [w_mid_price[i] == w_mid_price[i+1] for i in range(len(w_mid_price)-1)]
    e2_w = len(w_mid_price) -1 - sum(e1_w)
    exp_1_w = {'e1':{'cantidad': sum(e1_w), 'proporcion': np.round(sum(e1_w)/total_w,2)},'e2':{'cantidad': e2_w, 'proporcion': np.round(e2_w/total_w,2)},'total': len(w_mid_price)-1}
    df_exp_1_w = pd.DataFrame({'e1':{'cantidad': sum(e1_w), 'proporcion': np.round(sum(e1_w)/total_w,2)},'e2':{'cantidad': e2_w, 'proporcion': np.round(e2_w/total_w,2)},'total': len(w_mid_price)-1})
    # comprobar que tiene un comportamiento martingala:
    # imprimir los resultados:
    exp_1_w['e1']['proporcion']
    exp_1_w['e2']['proporcion']
    return df_exp_1_w
# ------ hacer el experimento 2 en función  -----
#primero crear un dataframe con el midprice:
df2_w = pd.DataFrame()
df2_w['Timestamp'] = None
df2_w = df2_w.assign(Midprices_w=None)
times = (l_ts)
ob_ts = list(data_ob.keys())
ob_imb = lambda v, d: np.sum(v.iloc[:d, 0]) / np.sum([v.iloc[:d, 0], v.iloc[:d, 1]])
mid = [(data_ob[ob_ts[i]]['ask'][0] + data_ob[ob_ts[i]]['bid'][0])* 0.5 for i in range(0, len(ob_ts))]
imbalance = [ob_imb(data_ob[i_ts][['bid_size', 'ask_size']], len(data_ob[i_ts])) for i_ts in ob_ts]
mids_w = list(np.array(mid) * np.array(imbalance))
df2_w['Timestamp'] = times
df2_w['Midprices_w'] = mids_w
df2_w = df2_w.set_index('Timestamp')
#...
dic_w = {}
for index, row in df2_w.iterrows():
    key = str(index.hour) + ":" + str(index.minute)
    value = row["Midprices_w"]
    try:
        dic_w[key].append(value)
    except KeyError:
        dic_w[key] = [value]
# ahora si hacer las iteraciones
metricasMin_w = {}
for i in list(dic_w): 
    e1_w = sum([dic_w[i][i_t] == dic_w[i][i_t+1] for i_t in range(len(dic_w[i])-1)])
    e2_w = sum([dic_w[i][i_t] != dic_w[i][i_t+1] for i_t in range(len(dic_w[i])-1)])
    metricasMin_w[i] = (
        {"e1_w" :{"cantidad_w" :  e1_w, "proporcion_w" : e1_w/(len(dic_w[i])-1)}, 
         "e2_w" :{"cantidad_w" :  e2_w, "proporcion_w" : e2_w/(len(dic_w[i])-1)},
         "total_w" : len(dic_w[i])-1
         }
        )
tot_w = []
for i in list(metricasMin_w):
    tot_w.append(metricasMin_w[i]["total_w"])
np.array(tot_w).sum()
# Proporción de frecuencia de martingalas
mgE1_w = []
for i in list(metricasMin_w):
    mgE1_w.append(metricasMin_w[i]["e1_w"]["cantidad_w"])
np.array(mgE1_w).sum()/np.array(tot_w).sum()
# Promedio de proporción de martingalas
propE1_w = []
for i in list(metricasMin_w):
    propE1_w.append(metricasMin_w[i]["e1_w"]["proporcion_w"])
np.array(propE1_w).mean()
#datos para llenar df fnal (experimento 2):
df_exp2_w = pd.DataFrame(metricasMin_w).T
df_e1_exp2_w=pd.DataFrame(df_exp2_w["e1_w"][i] for i in range(0,60))
df_e1_exp2_conteo_w = pd.DataFrame(df_e1_exp2_w["cantidad_w"])
df_e2_exp2_w=pd.DataFrame(df_exp2_w["e2_w"][i] for i in range(0,60))
df_e2_exp2_conteo_w = pd.DataFrame(df_e2_exp2_w["cantidad_w"])
df_e1_exp2_proporcion_w = pd.DataFrame(df_e1_exp2_w["proporcion_w"])
df_e2_exp2_proporcion_w = pd.DataFrame(1-df_e1_exp2_w["proporcion_w"])
df_exp2_total_w = pd.DataFrame(df_e1_exp2_conteo_w["cantidad_w"]+df_e2_exp2_conteo_w["cantidad_w"])
# llenar el df del experimento 2:
df_exp2_2_w = pd.DataFrame()
df_exp2_2_w = df_exp2_2_w.assign(e1_w=None)
df_exp2_2_w = df_exp2_2_w.assign(e2_w=None)
df_exp2_2_w = df_exp2_2_w.assign(total_w=None)
df_exp2_2_w = df_exp2_2_w.assign(proporcion1_w=None)
df_exp2_2_w = df_exp2_2_w.assign(proporcion2_w=None)
times = (l_ts)
valor_e1_w = df_e1_exp2_conteo_w 
valor_e2_w = df_e2_exp2_conteo_w 
valor_total_w = df_exp2_total_w
valor_proporcion1_w = df_e1_exp2_proporcion_w 
valor_proporcion2_w = df_e2_exp2_proporcion_w
df_exp2_2_w['e1_w'] = valor_e1_w
df_exp2_2_w['e2_w'] = valor_e2_w
df_exp2_2_w['total_w'] = valor_e1_w+valor_e2_w
df_exp2_2_w['proporcion1_w'] = valor_proporcion1_w
df_exp2_2_w['proporcion2_w'] = valor_proporcion2_w
#
#
#
#
#
# --------- ROLL MODEL ------- #

data_ob=dt.ob_data
ob_ts=list(data_ob.keys())#
l_ts= [pd.to_datetime(i_ts) for i_ts in ob_ts]#
l_mid = [(data_ob[ob_ts[i]]['ask'][0] + data_ob[ob_ts[i]]['bid'][0])* 0.5 for i in range(0, len(ob_ts))]
def roll_model(data,m_roll):
    data = data_ob
    ob_ts=list(data_ob.keys())#
    l_ts= [pd.to_datetime(i_ts) for i_ts in ob_ts]#
    l_mid = [(data_ob[ob_ts[i]]['ask'][0] + data_ob[ob_ts[i]]['bid'][0])* 0.5 for i in range(0, len(ob_ts))]
    m_roll = l_mid
    ob_ts = list(data.keys())
    bid = [data[ob_ts[i]]['bid'][0] for i in range(0, len(ob_ts))]
    ask = [data[ob_ts[i]]['ask'][0] for i in range(0, len(ob_ts))]
    spread = list(np.array(ask) - np.array(bid))
    df_roll = pd.DataFrame(index = pd.to_datetime(ob_ts), data = {'ask': ask,'ask_roll': ask,'bid': bid,'bid_roll': bid, 'mid': m_roll,'mid_roll': m_roll, 'spread': spread})
    delta_mid = df_roll['mid'].diff(1)
    delta_mid_lag = df_roll['mid'].shift(1).diff(1) 
    df_delta = pd.DataFrame({'delta mid t': delta_mid, 'delta_mid_lag': delta_mid_lag})
    covar = df_delta.cov().iloc[1, 0]
    constant = np.sqrt(-covar) 
    #crear df para llenar con los resultados: 
    df_roll['ask_roll'] = np.round(df_roll['mid'] + constant,3)   
    df_roll['bid_roll'] = np.round(df_roll['mid'] - constant,3)
    delta_mid_2 = df_roll['mid'].diff(1)
    delta_mid_lag_2 = df_roll['mid'].shift(1).diff(1) 
    df_delta_2 = pd.DataFrame({'delta mid t': delta_mid_2, 'delta_mid_lag': delta_mid_lag_2})
    covar = df_delta_2.cov().iloc[1, 0]
    constant = np.sqrt(-covar) 
    df_roll['mid_roll'] = np.round(df_roll['mid'] + constant,2)
    return df_roll
#head de los resultados roll:
roll_model2=roll_model(data_ob,l_mid).head(5)



#-------- GRAFICA PARA MIDPRICE POR MINUTO -----------#
def grafica_midprice(data: pd.DataFrame, price_type: str, colors: list) -> go.Figure:
    fig = go.Figure(data = [go.Bar(name = 'e1', x = data.index, y = data['proporcion1'], marker_color = colors[0]), \
        go.Bar(name = 'e2', x = data.index, y = data['proporcion2'], marker_color = colors[1])])
    fig.update_layout(autosize = False, width = 1000, height = 600, barmode = 'stack', \
        title_text = f'Proporción por minuto, midprice APT')
    fig.update_xaxes(title_text = 'Minuto')
    fig.update_yaxes(title_text = 'Proporción')
    return fig.show()
#grafica_midprice(data = fn.df_exp2_2, price_type = 'Mid Price', colors = ['orange', 'red'])

#-------- GRAFICA PARA WEIGHTED MIDPRICE POR MINUTO -----------#
def grafica_midprice_w(data: pd.DataFrame, price_type: str, colors: list)-> go.Figure:
    fig = go.Figure(data = [go.Bar(name = 'e1_w', x = data.index, y = data['proporcion1_w'],marker_color = colors[0]), \
        go.Bar(name = 'e2_w', x = data.index, y = data['proporcion2_w'], marker_color = colors[1])])
    fig.update_layout(autosize = False, width = 1000, height = 600, barmode = 'stack', \
        title_text = f'Proproción martingala por minuto, weighted midprice APT')
    fig.update_xaxes(title_text = 'Minuto')
    fig.update_yaxes(title_text = 'Proporción')
    return fig.show()
#grafica_midprice_w(data = fn.df_exp2_2_w, price_type = 'Mid Price', colors = ['orange', 'red'])





