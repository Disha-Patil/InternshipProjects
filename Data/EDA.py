#!/usr/bin/env python
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("bank_transactions.csv")


# In[3]:

print(data.shape)
# There are 1048567 observations and 9 columns

# In[4]:

data

# In[5]:

data.info()

# CustomerDOB, CustGender, CustLocation, CustAccountBalance have missing values

# In[6]:

print("Number of missing values for each column")
data.isnull().sum()
# In[7]:


#Dropping the "Transaction ID" column since it has no bearing on analysis
data.drop(["TransactionID"],axis=1,inplace=True)


# In[1]:


#Filling in missing values
custgender_mode = data.CustGender.mode().values[0]
data.CustGender.fillna(custgender_mode,inplace=True)

custlocation_mode = data.CustLocation.mode().values[0]
data.CustLocation.fillna(custlocation_mode,inplace=True)

custAB_median = data.CustAccountBalance.median()
data.CustAccountBalance.fillna(custAB_median,inplace=True)
# %%

# Splitting CustomerDOB, TransactionDate and TransactionTime into reading figures

data[["Birthdate", "Birthmonth", "Birthyear"]] = data["CustomerDOB"].str.split("/", expand = True)
data[["Tdate", "Tmonth", "Tyear"]] = data["TransactionDate"].str.split("/", expand = True)

time = []
for i in data.TransactionTime.values:
    hour = i//10000
    time.append(hour)

data["TransactionHour"] = time
# %%

#Dropping the below columns
data.drop(["CustomerDOB"],inplace=True,axis=1)
data.drop(["TransactionDate"],inplace=True,axis=1)
data.drop(["TransactionTime"],inplace=True,axis=1)
# %%

data

# %%

#Plotting Gender Distribution

gender = data['CustGender'].value_counts()
graph = gender.plot(kind='bar', color="c")
graph.set_title("Gender Distribution",)
graph.set_xlabel('Gender')
graph.set_ylabel('Number of People')

for rect in graph.patches:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2
    space = 1
    label = format(y_value)
    graph.annotate(label, (x_value, y_value), xytext=(0, space), textcoords="offset points", ha='center', va='bottom')
plt.show()

#Inference: There are a majority of male customers (73%), followed by female customers (27%) and transgender customers (1 customer)

# %%

#Calculating Age
data.Birthyear.fillna(data.Birthyear.median(),inplace=True)
age = []
difference = 0
for i in data.Birthyear.values:
    if int(i) < 16:
        difference = 16 - int(i)
    elif int(i) == 1800: #one observation with Birthyear 1800
        difference = 216
    else:
        difference = 100 - int(i) + 16
    age.append(difference)

data["Age"] = age
# %%

#Plotting Age Distribution

plt.hist(data.Age, bins = 10)
plt.title("Age Distribution")
plt.xlabel("Age Groups")
plt.ylabel("Number of people")
plt.show()

#Inference: The majority of customers are between 25-40 years of age. Those with ages > 200 are outliers, most likely with an error in their birthdate

# %%

#Plotting Location Distribution

location = data['CustLocation'].value_counts()
graph = location.plot(kind='bar', color="m")
graph.set_title("Location Distribution",)
graph.set_xlabel('Location')
graph.set_ylabel('Number of People')
plt.ylim(0,1000)
plt.show()

#Inference: The maximum number of transactions were made in New Delhi

# %%

#Plotting Distribution of Account Balance

graph = plt.hist(data.CustAccountBalance, bins = 10)
plt.title("Account Balance Distribution")
plt.xlabel("Amount of Balance")
plt.ylabel("Number of people")

plt.show()

#Inference: A majority of account balances were under Rs. 1.5 Crore (0.15 x 10^8) indicating that a majority of customers ranged from poor to fairly rich

# %%

#Plotting Transaction Amount Distribution
graph = plt.hist(data["TransactionAmount (INR)"], bins = 10, color = 'm')
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Number of people")

plt.show()

#Inference: Most transactions were under Rs. 2 lakhs (0.2 x 10^6) indicating low to relatively high transaction amounts

# %%

#Plotting Frequency of Transactions in each month

month = data['Tmonth'].value_counts()
graph = month.plot(kind='bar', color="y")
graph.set_title("Transaction Month Distribution",)
graph.set_xlabel('Month')
graph.set_ylabel('Number of People')

for rect in graph.patches:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2
    space = 1
    label = format(y_value)
    graph.annotate(label, (x_value, y_value), xytext=(0, space), textcoords="offset points", ha='center', va='bottom')
plt.show()

#Inference: August had the maximum number of transactions, followed by September and October. 
#In fact there were approximately 1/190th transactions in October as there were in August.
# %%

#Plotting Frequency of Transaction Hour

plt.figure(figsize=(20,10))
hour = data['TransactionHour'].value_counts()
graph = hour.plot(kind='bar', color="g")
graph.set_title("Transaction Hour Distribution",)
graph.set_xlabel('Hour')
graph.set_ylabel('Number of People')

for rect in graph.patches:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2
    space = 1
    label = format(y_value)
    graph.annotate(label, (x_value, y_value), xytext=(0, space), textcoords="offset points", ha='center', va='bottom')
plt.show()

# Inference: Most transactions occured late evening between 6-9 p.m.

# %%

#Plotting Transaction Distribution for each month

plt.figure(figsize=(20,10))

august = data[data.Tmonth == '8'].Tdate.value_counts()
graph1 = august.plot(kind='bar')
graph1.set_title("Transaction Distribution for August")
graph1.set_xlabel('Date')
graph1.set_ylabel('Number of People')

for rect in graph1.patches:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2
    space = 1
    label = format(y_value)
    graph1.annotate(label, (x_value, y_value), xytext=(0, space), textcoords="offset points", ha='center', va='bottom')

plt.show()

plt.figure(figsize=(20,10))
september = data[data.Tmonth == '9'].Tdate.value_counts()
graph2 = september.plot(kind='bar')
graph2.set_title("Transaction Distribution for September")
graph2.set_xlabel('Date')
graph2.set_ylabel('Number of People')

for rect in graph2.patches:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2
    space = 1
    label = format(y_value)
    graph2.annotate(label, (x_value, y_value), xytext=(0, space), textcoords="offset points", ha='center', va='bottom')

plt.show()

plt.figure(figsize=(20,10))
october = data[data.Tmonth == '10'].Tdate.value_counts()
graph3 = october.plot(kind='bar')
graph3.set_title("Transaction Distribution for October")
graph3.set_xlabel('Date')
graph3.set_ylabel('Number of People')

for rect in graph3.patches:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2
    space = 1
    label = format(y_value)
    graph3.annotate(label, (x_value, y_value), xytext=(0, space), textcoords="offset points", ha='center', va='bottom')

plt.show()

#Inference: Most transactions occurred in the middle of August, beginning of September, and on 21st October.
#There seems to be an error in the data recorded for October since the figure shows that transactions only occured on two days.
# %%

#Getting a statistics summary of all numeric categories
data.describe()
