
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project:High-Frequency Models a) Asset Pricing Theory b) Roll Model                                                   -- #
# -- script: data.py : python script for data collection                                                 -- #
# -- author: IvetteLandaverde                                                                       -- #
# -- license: GNU General Public License v3.0                                            -- #
# -- repository: https://github.com/IvetteLandaverde/Laboratorio-2--MyST                                                                  -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# ------- ORDER BOOKS -------- #

#dict_test = {'key_a': 'a', 'key_b': 'b'}
import numpy as np
import pandas as pd
import json

# Opening JSON file
f= open("orderbooks_05jul21.json")
#f=open('/Users/ivettelandaverde/Desktop/MyST/laboratorio 2/https:/github.com/IvetteLandaverde/Laboratorio-2--MyST/orderbooks_05jul21.json')
#f=open('https:/github.com/IvetteLandaverde/Laboratorio-2--MyST/orderbooks_05jul21.json')


# Returns JSON object as a dictionary
orderbooks_data = json.load(f)
ob_data = orderbooks_data["bitfinex"]

# Drop None keys
ob_data={i_key: i_value for i_key, i_value in ob_data.items() if i_value is not None}

# Convert to dataframe and rearange columns 
ob_data = {i_ob: pd.DataFrame(ob_data[i_ob])[["bid_size","bid","ask","ask_size"]] 
           if ob_data[i_ob] is not None else None for i_ob in list(ob_data.keys())}

# for largo 
i_count=0
l_data=[]
for i_data in ob_data.values():
    #i_data=list(ob_data.values())[0].vales()
    i_count += 1
    if i_data is None:
        print(i_data)
        l_data.append(i_count)

ob_data