#!/usr/bin/env python
# coding: utf-8

# In[86]:


import pandas as pd
import numpy as np
import plotly.express as px
import os
import plotly.graph_objs as go

import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns


# In[46]:





# In[47]:


data = pd.read_csv("cause_of_deaths.csv")
bevolking = pd.read_csv("world_population.csv")
GDP = pd.read_csv('GDP.csv')


# In[48]:


#We gebruiken alleen de bevolking van 2020
bevolking = bevolking.drop(columns=['2022 Population', '2015 Population', '2010 Population', '2000 Population', '1990 Population',
                                   '1980 Population', '1970 Population', 'Area (km²)', 'Density (per km²)', 'Growth Rate', 'World Population Percentage', 'Capital', 'Rank'])


# In[49]:


#we voegen de bevolking toe aan de originele dataset door te mergen op de Land Code
merged = pd.merge(data, bevolking, left_on='Code', right_on='CCA3')


# In[50]:





# In[51]:


#Van GDP gebruiken we alleen 2018
GDP2 = GDP.drop(GDP.iloc[:, 2:-2],axis = 1)
GDP2 = GDP2.drop(columns='2019')


# In[52]:


#We voegen de GDP van elk land toe
mergedGDP = pd.merge(merged, GDP2, left_on='Code', right_on='Country Code')


# In[42]:





# In[55]:


#We maken ook een dataset met het percentage van de bevolking dat dat jaar is overleden door die oorzaak ipv het totaal aantal doden
procent = merged[['Meningitis', "Alzheimer's Disease and Other Dementias", "Parkinson's Disease", "Nutritional Deficiencies", "Malaria",
               "Drowning", "Interpersonal Violence", "Maternal Disorders", "HIV/AIDS", "Drug Use Disorders", "Tuberculosis", "Cardiovascular Diseases",
               "Lower Respiratory Infections", "Neonatal Disorders", "Alcohol Use Disorders", "Self-harm", "Exposure to Forces of Nature",
               "Diarrheal Diseases", "Environmental Heat and Cold Exposure", "Neoplasms", "Conflict and Terrorism"
              ]].div(merged['2020 Population'], axis=0)


# In[68]:


procent['Code'] = merged['Code']
procent['Country/Territory_x'] = merged['Country/Territory_x']
procent['Year'] = merged['Year']


# In[59]:





# In[60]:


#We maken een nieuwe dataframe waar we de landen groeperen per continent en jaar
Group = merged.groupby(['Continent', 'Year'], as_index=False)


# In[61]:





# In[69]:


procent2 = procent.copy()


# In[70]:


procentGDP = pd.merge(procent2, GDP2, left_on='Code', right_on='Country Code')


# In[71]:





# In[72]:


procentGDP2018 = procentGDP.loc[procentGDP['Year']==2018]


# In[ ]:


st.title('Statistische voorspellingen')


# In[73]:


#fig = plt.figure()
#sns.regplot(data= data3, x="Total_Deaths", y="2020 Population", ci=None)
#sns.scatterplot(data= data3, x="Total_Deaths", y="2020 Population",color="red",marker="s")
#plt.title('Regressiemodel geplot aantal doden wordt voorspeld door de populatie')
#plt.xlabel('Aantal doden per land x miljoen')
#plt.ylabel('Populatie per land x 100 miljoen')
#plt.show()
#st.plotly_chart(fig, use_container_width=True)


# In[87]:


fig = plt.figure()
sns.regplot(data= procentGDP2018, x="Drug Use Disorders", y="2018", ci=None)
sns.scatterplot(data= procentGDP2018, x="Drug Use Disorders", y="2018",color="red",marker="s")
plt.title('Regressiemodel Aantal doden voorspelt door BBP')
plt.xlabel('Percentage van bevolking overleden door Drug Use Disorders')
plt.ylabel('BBP')
plt.show()
st.plotly_chart(fig, use_container_width=True)


# In[89]:


fig = plt.figure()
sns.regplot(data= procentGDP2018, x="Nutritional Deficiencies", y="2018", ci=None)
sns.scatterplot(data= procentGDP2018, x="Nutritional Deficiencies", y="2018",color="red",marker="s")
plt.title('Regressiemodel Aantal doden voorspelt door BBP')
plt.xlabel('Percentage van bevolking overleden door Nutritional Deficiencies')
plt.ylabel('BBP')
plt.show()
st.plotly_chart(fig, use_container_width=True)


# In[92]:


fig = plt.figure()
sns.regplot(data= procentGDP2018, x="Neonatal Disorders", y="2018", ci=None)
sns.scatterplot(data= procentGDP2018, x="Neonatal Disorders", y="2018",color="red",marker="s")
plt.title('Regressiemodel Aantal doden voorspelt door BBP')
plt.xlabel('Percentage van bevolking overleden door Neonatal Disorders')
plt.ylabel('BBP')
plt.show()
st.plotly_chart(fig, use_container_width=True)


# In[ ]:




