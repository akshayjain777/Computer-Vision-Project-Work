# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 16:40:09 2020

@author: akshay.jain23
"""

import pandas as pd
import matplotlib as plt

df= pd.read_csv("TrackNet3cv5.csv",index_col=0)

df.columns=['fn','x','y']

print(df.head())

df.plot(x='fn', y='y', style='o')


plt.pyplot.plot(df.fn[-70:],df.y[-70:],'bo')