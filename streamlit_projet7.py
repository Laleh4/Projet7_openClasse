# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 14:49:57 2022

@author: laleh
"""

import streamlit as st
import pandas as pd
import pathlib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
import shap

import gdown
import os
import shutil


TT=pathlib.Path(__file__).parent.resolve()
path_L=str(TT).replace("\\","/") 
st.title("Credit Analysis  App")
image = Image.open(path_L+"/money_image.jpg")

st.image(image,width=500)

f=open(path_L+'/data_projet7/data_Milestone3.pkl','rb')
val_final=pickle.load(f)
seuil_final=pickle.load(f)
seuil_final=seuil_final[0,0]
rf_grid=pickle.load(f)
rf_model = rf_grid.best_estimator_
f.close()

X_trains=pd.read_csv(path_L+"/data_projet7/X_train.csv")
X_valids=pd.read_csv(path_L+"/data_projet7/X_valid.csv")
y_trains=pd.read_csv(path_L+"/data_projet7/y_train.csv")
y_valids=pd.read_csv(path_L+"/data_projet7/y_valid.csv")

#read_csv reads the index column
X_trains=X_trains.iloc[:,1:]
X_valids=X_valids.iloc[:,1:]
y_trains=y_trains.iloc[:,1:]
y_valids=y_valids.iloc[:,1:]
columns_titles=list(X_trains.columns)
 

    
menu=["Home", "General","Analysis"]
choice=st.sidebar.selectbox("Menu",menu)

if choice=="General":
        st.subheader("General about feature importance of the model")      
        importances_rf= pd.DataFrame({
             'Attribute': columns_titles,
             'Importance': np.abs(rf_model.feature_importances_)
         })
 
        importances_rf=importances_rf.sort_values(by='Importance', ascending=False)
        list_Feat_imp=list(importances_rf["Attribute"])
        
        fig, ax = plt.subplots()
        cmap = plt.cm.coolwarm
        ax.bar(x=importances_rf.Attribute,height=importances_rf.Importance,width=0.2,color=cmap(np.linspace(0,1,len(importances_rf)))) #['r','b','g'])
        plt.xticks(rotation=90)
        st.pyplot(fig)
       
        url="https://drive.google.com/drive/folders/10UoYyWKnDCwwvzk8NMWAE-5XNarOSegA" #?usp=sharing
        gdown.download_folder(url, quiet=True)      
        TT=pathlib.Path("__file__").parent.resolve()
        path_L=str(TT).replace("\\","/") 
        P=pathlib.PureWindowsPath(path_L)
        path_orig=str(P) #str(P.parents[0])
        
        path_dest=path_orig+"/dossier_test_copie2/" 
        if not os.path.exists(path_dest):
            os.mkdir(path_dest)
        shutil.move(path_orig+"\keras_metadata.pb", path_dest+"\keras_metadata.pb")
        shutil.move(path_orig+"\saved_model.pb", path_dest+"\saved_model.pb")







if choice=="Analysis":
        st.subheader("Analysis of your case")
        i_client = st.number_input('Insert the client number in data base (between 0 and 2000)',min_value= 0,max_value=2000)
        y_pred= rf_model.predict_proba(X_valids.loc[[i_client]])[0,1]

        
        if y_pred<seuil_final:
            st.markdown("Your credit is **approved.**")
        else:
            st.write("Your credit is **refused.**")            
        
        importances_rf= pd.DataFrame({
             'Attribute': columns_titles,
             'Importance': np.abs(rf_model.feature_importances_)
         })

        importances_rf=importances_rf.sort_values(by='Importance', ascending=False)
        list_Feat_imp=list(importances_rf["Attribute"])
        y_pred= rf_model.predict_proba(X_valids)[:, 1]
        y_pred_bin=[1 if y>seuil_final else 0 for y in y_pred]

  
        list_mins=[]; list_maxs=[]; list_yours=[]
        for tt in list_Feat_imp:
            vv=[X_valids[tt][i] if y_pred_bin[i]==0 else np.nan for i in range(len(y_pred_bin))]
            temp=pd.DataFrame(vv)
            describe_temp=temp.describe()
            list_mins.append(describe_temp.loc["min"].values[0]) 
            list_maxs.append(describe_temp.loc["max"].values[0])
            list_yours.append(X_valids[tt][i_client])
        importances_rf["Min value"]=list_mins
        importances_rf["Max value"]=list_maxs
        importances_rf["Your value"]=list_yours
        
        fig, ax = plt.subplots()
        ax=plt.bar(x=importances_rf.Attribute,height=list_maxs,width=0.2)
        ax=plt.bar(x=importances_rf.Attribute,height=list_mins,color='r',width=0.2)
        ax=plt.bar(x=importances_rf.Attribute,height=list_yours,color='k',width=0.2)
        plt.xticks(rotation=90);
        plt.grid()
        #plt.xlabel("Feautures of the model in Importance order ")
        plt.ylabel("Values  ")
        plt.title('Comparing your values (black)  with those of accepted credit people')
        st.pyplot(fig)
        
        
        explainer = shap.TreeExplainer(rf_model,feature_dependence="independent")
        
        
        choosen_instance = X_valids.loc[[i_client]]
        shap_values = explainer.shap_values(choosen_instance)
        index_sort_zero=np.argsort(explainer.shap_values(X_valids.iloc[i_client,:], y_pred_bin[i_client])[0]) # cotè rouge: point faibles qui réduit
        index_sort_one=np.argsort(explainer.shap_values(X_valids.iloc[i_client,:], y_pred_bin[i_client])[1])
        
        
        
        col1,col2=st.beta_columns(2)
        with col1:
                with st.beta_expander("Your 5 strengths"):
                    st.write(list(np.array(columns_titles)[index_sort_one[:5]])[0])
                    st.write(list(np.array(columns_titles)[index_sort_one[:5]])[1])
                    st.write(list(np.array(columns_titles)[index_sort_one[:5]])[2])
                    st.write(list(np.array(columns_titles)[index_sort_one[:5]])[3])
                    st.write(list(np.array(columns_titles)[index_sort_one[:5]])[4])
                    
                    
        with col2:
                with st.beta_expander("Your 5 weaknesses"):
                    st.write(list(np.array(columns_titles)[index_sort_zero[:5]])[0])
                    st.write(list(np.array(columns_titles)[index_sort_zero[:5]])[1])
                    st.write(list(np.array(columns_titles)[index_sort_zero[:5]])[2])
                    st.write(list(np.array(columns_titles)[index_sort_zero[:5]])[3])
                    st.write(list(np.array(columns_titles)[index_sort_zero[:5]])[4])
        
        fig, ax = plt.subplots()
       

        plt.scatter(range(len(y_pred)),1-y_pred)
        plt.plot([0, len(y_pred)],[1-rf_model.predict_proba(choosen_instance)[0][1],1-rf_model.predict_proba(choosen_instance)[0][1]],color='k')
        plt.text(0,1-rf_model.predict_proba(choosen_instance)[0][1],"Your score", fontsize=30.
                )
        plt.plot([0, len(y_pred)],[1-seuil_final,1-seuil_final],color="r")
        plt.text(0,1-seuil_final-0.02
                 ,"Min  score", fontsize=20,color='r')
        plt.xlabel("Others")
        plt.ylabel("Scores")
        plt.title("COMPARING YOUR SCORE WITH OTHER'S AND WITH THE  MIN ACCEPTED CREDIT VALUE")
                
        st.pyplot(fig)