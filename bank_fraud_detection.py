#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import json
def main():
    
    with open('datasets/transactions.txt') as fp: # read line from file
        while True:
            line=fp.readline()
            if not line:
                break
            line= json.loads(line) #line as dictionary
      
            write_csv(line)   

def write_csv(data):
    with open("transaction.csv",'a') as file:
        writer=csv.writer(file)
        writer.writerow([data['accountNumber'],data['customerId'],data['creditLimit'],
                        data['availableMoney'],data['transactionDateTime'],data['transactionAmount'],
                        data['merchantName'],data['acqCountry'],data['merchantCountryCode'],data['posEntryMode'],
                        data['posConditionCode'],data['merchantCategoryCode'],data['currentExpDate'],
                        data['accountOpenDate'],data['dateOfLastAddressChange'],data['cardCVV'],data['enteredCVV'],
                        data['cardLast4Digits'],data['transactionType'],data['echoBuffer'],data['currentBalance'],
                        data['merchantCity'],data['merchantState'],data['merchantZip'],data['cardPresent'],
                        data['posOnPremises'],data['recurringAuthInd'],data['expirationDateKeyInMatch'],
                        data['isFraud']])

if __name__=='__main__':
    main()                           


# In[ ]:


#add header to csv file
import pandas as pd
import json

s=["accountNumber", "customerId", "creditLimit", "availableMoney", "transactionDateTime", "transactionAmount", "merchantName", "acqCountry", "merchantCountryCode", "posEntryMode", "posConditionCode", "merchantCategoryCode", "currentExpDate", "accountOpenDate", "dateOfLastAddressChange", "cardCVV", "enteredCVV", "cardLast4Digits", "transactionType", "echoBuffer", "currentBalance", "merchantCity", "merchantState", 'merchantZip', "cardPresent", "posOnPremises", "recurringAuthInd", "expirationDateKeyInMatch", "isFraud"]
data=pd.read_csv("transactions.csv",header=None)
data.to_csv("transaction.csv",header=s,index=False)


# In[ ]:





# In[1]:


import pandas as pd
df=pd.read_csv("datasets/transactions.csv",index_col=False)


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df.describe()


# In[4]:


df.info()


# In[3]:


#df['isFraud']=df['isFraud'].astype(int)
#df['cardPresent']=df['cardPresent'].astype(int)
#df['expirationDateKeyInMatch']=df['expirationDateKeyInMatch'].astype(int)
#df.drop(['echoBuffer','merchantCity','merchantState','merchantZip','posOnPremises','recurringAuthInd'],axis=1,inplace=True)
#df.drop(['transactionDateTime','merchantName','acqCountry','merchantCountryCode','posEntryMode','posConditionCode'],axis=1,inplace=True)
#df.drop(['currentExpDate','accountOpenDate','dateOfLastAddressChange'],axis=1,inplace=True)
#df.drop(['transactionType'],axis=1,inplace=True)
df.drop(['accountNumber','customerId','cardCVV','enteredCVV','cardLast4Digits'],axis=1,inplace=True)


# In[34]:


df.head()


# In[4]:


from sklearn.preprocessing import StandardScaler 


# In[5]:


scaler=StandardScaler()

amount = df['creditLimit'].values

df['creditLimit'] = scaler.fit_transform(amount.reshape(-1, 1))


# In[6]:


amount = df['availableMoney'].values

df['availableMoney'] = scaler.fit_transform(amount.reshape(-1, 1))


# In[7]:


amount = df['transactionAmount'].values

df['transactionAmount'] = scaler.fit_transform(amount.reshape(-1, 1))


# In[8]:


amount = df['currentBalance'].values

df['currentBalance'] = scaler.fit_transform(amount.reshape(-1, 1))


# In[9]:


df.head()


# In[38]:


df.corr()


# In[30]:


sns.pairplot(df)


# In[19]:


sns.heatmap(df,cmap="YlGnBu")


# In[20]:


plt.scatter(df['availableMoney'],df['currentBalance'],c=y)
plt.show()


# In[10]:


y=df['isFraud']
df.drop('isFraud',axis=1,inplace=True)
x=df


# In[11]:


df.info()


# In[12]:


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from sklearn.metrics import accuracy_score,roc_auc_score,precision_score, recall_score,f1_score,classification_report


# In[13]:


seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(x, y ,test_size=test_size, random_state=seed)


# In[14]:


model = XGBClassifier()
model.fit(X_train, y_train)


# In[15]:


y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# In[16]:


accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:




