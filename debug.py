# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 10:53:13 2022

@author: laleh
"""

import streamlit as st
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax=plt.bar(x=[0,1,2],height=[4,-6,8],width=0.2)
st.pyplot(fig)