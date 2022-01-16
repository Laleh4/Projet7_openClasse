# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 14:49:57 2022

@author: laleh
"""

import streamlit as st
import pandas as pd
import pathlib

TT=pathlib.Path(__file__).parent.resolve()
path_L=str(TT).replace("\\","/") 
#df=pd.read_csv(path_L+"/labels.csv") 
df_train=pd.read_csv(path_L+"/X_train.csv")
#dft=df[df.name==image_file.name]
#if len(dft):
#    return str(dft["label"].values[0])   
#else:
#    return " "

st.write(df_train["EXT_SOURCE_3"][0])