#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import plotly as plt
import plotly.express as px
import os
import plotly.graph_objs as go
import streamlit as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols


# In[2]:


data = pd.read_csv("https://raw.githubusercontent.com/JuliaaaaaaZ/hallo/main/cause_of_deaths.csv")


# In[3]:


bevolking = pd.read_csv("https://raw.githubusercontent.com/JuliaaaaaaZ/hallo/main/world_population.csv")


# In[4]:


#data.head()


# In[5]:


#bevolking.head()


# In[6]:


#Alleen bevolking uit 2020 houden
bevolking = bevolking.drop(columns=['2022 Population', '2015 Population', '2010 Population', '2000 Population', '1990 Population',
                                   '1980 Population', '1970 Population', 'Area (km²)', 'Density (per km²)', 'Growth Rate', 'World Population Percentage', 'Capital', 'Rank'])


# In[7]:


#bevolking


# In[ ]:





# In[ ]:





# In[8]:


dataDeaths = data.drop(columns=["Country/Territory", "Code", "Year"])
dataoorzaak = dataDeaths.idxmax(axis=1)
data["Top Cause"] = dataoorzaak
#data.head()


# In[9]:


#data.head()


# In[10]:


fig = px.choropleth(data,               
              locations="Code",               
              color="Top Cause",
              hover_name="Country/Territory",  
              animation_frame="Year",
              width=800      
)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

st.plotly_chart(fig)


# In[11]:


fig = px.scatter(  data_frame=data,  y='Malaria',   x='HIV/AIDS',  color='Code', 
                 animation_frame='Year')

fig['layout'].pop('updatemenus')

st.plotly_chart(fig)


# In[12]:


#Alleen jaar 2019
data.loc[data['Year'] == 2019]


# In[13]:


#copy maken van data om kolommen te vervagen door percentages 
data2 = data.copy()


# In[14]:


#jaar 2019 selecteren
data2 = data2.loc[data['Year'] == 2019]


# In[15]:


data2['Total_Deaths']= data2.sum(axis=1)
#data2


# In[16]:


data3 = pd.merge(data2, bevolking, left_on='Code', right_on='CCA3')


# In[17]:


#data3.info()


# In[18]:


data3=data3[data3["Total_Deaths"]<8000000]


# In[19]:


fig = px.choropleth(data3,               
              locations="Code",               
              color="Total_Deaths",
              hover_name="Country/Territory_x",  
              animation_frame="Year",
              width=800      
)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

st.plotly_chart(fig)


# In[20]:


fig = px.scatter(data3, x="Total_Deaths", y="2020 Population")
fig.show()


# In[21]:


sns.regplot(data= data3, x="Total_Deaths", y="2020 Population",ci=None)


# In[23]:


fig = plt.figure()
sns.regplot(data= data3, x="Total_Deaths", y="2020 Population", ci=None)
sns.scatterplot(data= data3, x="Total_Deaths", y="2020 Population",color="red",marker="s")
plt.title('Regressiemodel geplot aantal doden wordt voorspeld door de populatie')
plt.xlabel('Aantal doden per land x miljoen')
plt.ylabel('Populatie per land x 100 miljoen')
st.pyplot()


# In[ ]:


#deaths_population = ols("Total_Deaths~ 2020 Population", data=data3).fit()
#print(model.params)


# In[24]:


Y= data3['Total_Deaths']
X= data3['2020 Population']
model = sm.OLS(Y, X).fit()
predictions = model.predict(Y) 

print_model = model.summary()
print(print_model)


# In[25]:


y= data3['Total_Deaths']
X= sm.add_constant(data3['2020 Population'])
model = sm.OLS(y, X).fit()

predictions = model.predict() 

model.summary()


# In[26]:


#Aantal doden per oorzaak delen door bevolking waardoor we een fractie krijgen.
data5 = data3[['Meningitis', "Alzheimer's Disease and Other Dementias", "Parkinson's Disease", "Nutritional Deficiencies", "Malaria",
               "Drowning", "Interpersonal Violence", "Maternal Disorders", "HIV/AIDS", "Drug Use Disorders", "Tuberculosis", "Cardiovascular Diseases",
               "Lower Respiratory Infections", "Neonatal Disorders", "Alcohol Use Disorders", "Self-harm", "Exposure to Forces of Nature",
               "Diarrheal Diseases", "Environmental Heat and Cold Exposure", "Neoplasms", "Conflict and Terrorism", "Total_Deaths"
              ]].div(data3['2020 Population'], axis=0)


# In[27]:


#Landen en landcode toevoegen
data5['Code'] = data3['Code']
data5['Country/Territory_x'] = data3['Country/Territory_x']
data5['Continent'] = data3['Continent']
data5['Top Cause'] = data3["Top Cause"]


# In[ ]:


#data5


# In[28]:


df = px.data.gapminder().query("year==2007")
df = df.rename(columns=dict(pop="Population",
                            gdpPercap="GDP per Capita",
                            lifeExp="Life Expectancy"))
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
       locations=data5['Code'], # Spatial coordinates
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
fig.update_layout(title=f"<b>{first_title}</b>",title_x=0.5)

st.plotly_chart(fig)


# In[29]:


data5= data5[data5['Total_Deaths']<1]
fig = px.box(data5, x="Continent", y="Total_Deaths", color="Continent", title='Boxplot totaal aantal doden in 2019 per continent')

st.plotly_chart(fig)


# In[30]:


fig = px.box(data5, x="Continent", y="Total_Deaths", color="Top Cause", title='Boxplot totaal aantal doden in 2019 per continent met de verschillende oorzaken')

st.plotly_chart(fig)


# In[31]:


fig = px.histogram(data5, x="Total_Deaths", color="Top Cause", title= 'Histogram van het aantal landen dat doden heeft per ziekte')

fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)


st.plotly_chart(fig)


# In[32]:


fig = px.scatter(data5, x="HIV/AIDS", y='Malaria', title='Scatterplot verhouding tussen totaal aantal doden bij Malaria en Hiv of Aids')
st.plotly_chart(fig)


# In[33]:


st.title("Selecteer een titel voor visualisatie")
line = st.selectbox("Kies een continent:", data5['Continent'])
oorzaak_dood = "Top Cause"
continent = "Continent"
kleurkeuze = st.radio("Visualisatie op baseren op:", [oorzaak_dood, continent])
fig = px.box(data5[data5['Continent']==line], x="Top Cause", y="Total_Deaths", color=kleurkeuze
                ,title="Spreiding aantal doden in %s met de belangrijkste dood oorzaken"%(line))
st.plotly_chart(fig)


# In[ ]:




