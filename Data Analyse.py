#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import os
import plotly.graph_objs as go
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:





# In[3]:


data = pd.read_csv("cause_of_deaths.csv")
bevolking = pd.read_csv("world_population.csv")
GDP = pd.read_csv('GDP.csv')


# In[4]:


bevolking = bevolking.drop(columns=['2022 Population', '2015 Population', '2010 Population', '2000 Population', '1990 Population',
                                   '1980 Population', '1970 Population', 'Area (km²)', 'Density (per km²)', 'Growth Rate', 'World Population Percentage', 'Capital', 'Rank'])


# In[5]:


merged = pd.merge(data, bevolking, left_on='Code', right_on='CCA3')


# In[ ]:





# In[6]:


GDP2 = GDP.drop(GDP.iloc[:, 2:-2],axis = 1)
GDP2 = GDP2.drop(columns='2019')


# In[7]:


mergedGDP = pd.merge(merged, GDP2, left_on='Code', right_on='Country Code')


# In[8]:


procent = merged[['Meningitis', "Alzheimer's Disease and Other Dementias", "Parkinson's Disease", "Nutritional Deficiencies", "Malaria",
               "Drowning", "Interpersonal Violence", "Maternal Disorders", "HIV/AIDS", "Drug Use Disorders", "Tuberculosis", "Cardiovascular Diseases",
               "Lower Respiratory Infections", "Neonatal Disorders", "Alcohol Use Disorders", "Self-harm", "Exposure to Forces of Nature",
               "Diarrheal Diseases", "Environmental Heat and Cold Exposure", "Neoplasms", "Conflict and Terrorism"
              ]].div(merged['2020 Population'], axis=0)


# In[9]:


procent['Code'] = merged['Code']
procent['Country/Territory_x'] = merged['Country/Territory_x']
procent['Year'] = merged['Year']


# In[10]:


procent2 = procent.copy()


# In[11]:


procentGDP = pd.merge(procent2, GDP2, left_on='Code', right_on='Country Code')


# In[12]:


procentGDP2018 = procentGDP.loc[procentGDP['Year']==2018]


# In[13]:


dataDeaths = data.drop(columns=["Country/Territory", "Code", "Year"])
dataoorzaak = dataDeaths.idxmax(axis=1)
data["Top Cause"] = dataoorzaak


# In[15]:


data2 = data.copy()


# In[16]:


data2 = data2.loc[data['Year'] == 2019]


# In[17]:


data2['Total_Deaths']= data2.sum(axis=1)


# In[18]:


data3 = pd.merge(data2, bevolking, left_on='Code', right_on='CCA3')


# In[19]:


data3 =data3[data3["Total_Deaths"]<8000000]


# In[ ]:


st.subheader('Kaart van totaal aantal doden')


# In[20]:


fig = px.choropleth(data3,               
              locations="Code",               
              color="Total_Deaths",
              hover_name="Country/Territory_x",  
              animation_frame="Year",
              width=800      
)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
st.plotly_chart(fig, use_container_width=True)


# In[34]:


st.subheader('Scatterplot van totaal aantal doden tegenover de populatie')


# In[21]:


fig = px.scatter(data3, x="Total_Deaths", y="2020 Population")
fig.show()
st.plotly_chart(fig, use_container_width=True)


# In[22]:


data5 = data3[['Meningitis', "Alzheimer's Disease and Other Dementias", "Parkinson's Disease", "Nutritional Deficiencies", "Malaria",
               "Drowning", "Interpersonal Violence", "Maternal Disorders", "HIV/AIDS", "Drug Use Disorders", "Tuberculosis", "Cardiovascular Diseases",
               "Lower Respiratory Infections", "Neonatal Disorders", "Alcohol Use Disorders", "Self-harm", "Exposure to Forces of Nature",
               "Diarrheal Diseases", "Environmental Heat and Cold Exposure", "Neoplasms", "Conflict and Terrorism", "Total_Deaths"
              ]].div(data3['2020 Population'], axis=0)


# In[23]:


data5['Code'] = data3['Code']
data5['Country/Territory_x'] = data3['Country/Territory_x']
data5['Continent'] = data3['Continent']
data5['Top Cause'] = data3["Top Cause"]


# In[24]:


data5= data5[data5['Total_Deaths']<1]
fig = px.box(data5, x="Continent", y="Total_Deaths", color="Continent", title='Boxplot totaal aantal doden in 2019 per continent')

fig.show()
st.plotly_chart(fig, use_container_width=True)


# In[25]:


fig = px.box(data5, x="Continent", y="Total_Deaths", color="Top Cause", title='Boxplot totaal aantal doden in 2019 per continent met de verschillende oorzaken')

fig.show()
st.plotly_chart(fig, use_container_width=True)


# In[26]:


fig = px.histogram(data5, x="Total_Deaths", color="Top Cause", title= 'Histogram van het aantal landen dat doden heeft per ziekte')

fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)

fig.show()
st.plotly_chart(fig, use_container_width=True)


# In[27]:


st.title("Selecteer een titel voor visualisatie")
line = st.selectbox("Kies een continent:", data5['Continent'])
oorzaak_dood = "Top Cause"
continent = "Continent"
kleurkeuze = st.radio("Visualisatie op baseren op:", [oorzaak_dood, continent])
fig = px.box(data5[data5['Continent']==line], x="Top Cause", y="Total_Deaths", color=kleurkeuze
                ,title="Spreiding aantal doden in %s met de belangrijkste dood oorzaken"%(line))
st.plotly_chart(fig, use_container_width=True)
fig.show()
st.plotly_chart(fig, use_container_width=True)


# In[28]:


st.subheader('Grootste doodsoorzaak door de jaren heen')
fig = px.choropleth(data,               
              locations="Code",               
              color="Top Cause",
              hover_name="Country/Territory",  
              animation_frame="Year",
              width=800      
)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
st.plotly_chart(fig, use_container_width=True)


# In[32]:


merged = merged.loc[data['Year'] == 2019]


# In[33]:


st.subheader('Kaart met slachtoffers per oorzaak')
cols_dd = ['Meningitis', "Alzheimer's Disease and Other Dementias", "Parkinson's Disease", "Nutritional Deficiencies", "Malaria",
               "Drowning", "Interpersonal Violence", "Maternal Disorders", "HIV/AIDS", "Drug Use Disorders", "Tuberculosis", "Cardiovascular Diseases",
               "Lower Respiratory Infections", "Neonatal Disorders", "Alcohol Use Disorders", "Self-harm", "Exposure to Forces of Nature",
               "Diarrheal Diseases", "Environmental Heat and Cold Exposure", "Neoplasms", "Conflict and Terrorism"
              ]
# we need to add this to select which trace 
# is going to be visible
visible = np.array(cols_dd)

# define traces and buttons at once
traces = []
buttons = []
for value in cols_dd:
    traces.append(go.Choropleth(
       locations=merged['Code'], # Spatial coordinates
        z=data5[value].astype(float), # Data to be color-coded
        colorbar_title=value,
        visible= True if value==cols_dd[0] else False))

    buttons.append(dict(label=value,
                        method="update",
                        args=[{"visible":list(visible==value)},
                              {"title":f"<b>{value}</b>"}]))

updatemenus = [{"active":0,
                "buttons":buttons,
               }]
fig = go.Figure(data=traces,
                layout=dict(updatemenus=updatemenus))
# This is in order to get the first title displayed correctly
first_title = cols_dd[0]
fig.update_layout(title=f"<b>{first_title}</b>",title_x=0.5, width=800)
fig.show()
st.plotly_chart(fig, use_container_width=True)

