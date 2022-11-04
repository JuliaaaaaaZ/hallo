#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import streamlit as st


# In[2]:





# In[5]:


data = pd.read_csv("cause_of_deaths.csv")
bevolking = pd.read_csv("world_population.csv")
GDP = pd.read_csv('GDP.csv')


# In[ ]:


st.header('Dataset Inspectie')


# In[ ]:


st.subheader('Dataset doodsoorzaken')


# In[12]:


data


# In[ ]:


st.subheader('hulpbron 1: Wereldbevolking')


# In[10]:


bevolking


# In[ ]:


st.subheader('Hulpbron 2: BBP per land')


# In[8]:


GDP


# In[ ]:




