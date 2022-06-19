
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: visualizations.py : python script with data visualization functions                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                               -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import data as dt
import numpy as np
import pandas as pd
import functions as fn
import plotly.graph_objects as go

#
# basic plotly plot
plot_data_1 = go.Figure(go.Bar(x=fn.df_exp2_2['e1'], y=fn.df_exp2_2['proporcion1']))
plot_data_1.show()

# basic plotly plot
plot_data_2 = go.Figure(go.Bar(x=fn.df_exp2_2_w['e1_w'], y=fn.df_exp2_2_w['proporcion1_w']))
plot_data_2.show()
#



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