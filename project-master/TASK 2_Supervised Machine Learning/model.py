#!/usr/bin/env python
# coding: utf-8

# ## **Task 2: To Explore Supervised Machine Learning**

# The Goal is to Understand how a Machine Learning Algorithm works. In this notebook, we will explore the Given Data set consisting of two columns. We will build a Simple linear Regression Model to predict the Percentage score obtained by a student aaccording to the number of hours they study.

# ### Author: Rohan Kamble
# 

# In[1]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Reading data from remote link
url = "http://bit.ly/w-data"
df = pd.read_csv(url)
df.head()


X = df.iloc[:, :-1].values  #Independent variable
y = df.iloc[:, 1].values  #Target variable


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

# Predicting for test data
pred = regressor.predict(X_test) # Predicting the scores
if pred>=100:
    print("You are Perfect! You have scored 100%")
else:
    print("Predicted Percentage Score = {}".format(pred[0]))
    
# Saving model to disk
import pickle
model = pickle.dump(regressor,open('model.pkl','wb'))

# loading model

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[9.25]]))
