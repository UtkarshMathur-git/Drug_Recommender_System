# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 20:18:05 2021

@author: Utkarsh Mathur
"""

import streamlit as st 
import pickle
import pandas as pd

def recommend(condition):
    drug_index = drugs[drugs['condition'] == condition].index[0]
    distances = similarity[drug_index]
    drug_list = sorted((list(enumerate(distances))), reverse=True, key=lambda x:x[1])[1:6]
    
    recommended_drugs = []
    for i in drug_list:
        recommended_drugs.append(drugs.iloc[i[0]].drugName)
    return recommended_drugs

drugs_list = pickle.load(open('drugs_dict.pkl','rb'))
drugs = pd.DataFrame(drugs_list)

similarity = pickle.load(open('similarity.pkl','rb'))

st.title("Drug Recommender System")

selected_drug_name = st.selectbox(
"Please select health condition and get medicine recommendation",
drugs['condition'].values)

if st.button('Recommend'):
    recommendation = recommend(selected_drug_name)
    for i in recommendation:
        st.write(i)