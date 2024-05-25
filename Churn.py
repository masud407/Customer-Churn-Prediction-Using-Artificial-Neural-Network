#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[2]:


df = pd.read_csv("customer_churn.csv")
df.sample(5)


# In[3]:


#Check the data types
df.dtypes


# In[4]:


df.drop('customerID',axis='columns',inplace=True) #drop customer ID


# In[5]:


df.TotalCharges.values  #check the datatype, need the numerical values as float type (numeric)


# In[6]:


df.MonthlyCharges.values


# In[7]:


pd.to_numeric(df.TotalCharges)

#can't be executed as there are some empty spaces(for example: "NA'") in that column


# In[8]:


#the below line of code is essentially checking if the 'TotalCharges' column contains non-numeric values

pd.to_numeric(df.TotalCharges,errors='coerce').isnull()  
#The errors='coerce' parameter tells pandas to coerce errors to NaN (Not a Number) values if any conversion error occurs.
#.isnull(): After converting the values to numeric format, this code checks for null values (for example: row 488 is null).


# In[9]:


df.shape


# In[10]:


# 488th row for TotalCharges has nothing, to check this

df.iloc[488].TotalCharges


# In[11]:


#Assigning a new matrix containing available values for TotalCharges

df1 = df[df.TotalCharges!=' ']
df1.shape  #11 rows having 'NA' are deleted


# In[12]:


df1.dtypes


# In[13]:


df1.TotalCharges=pd.to_numeric(df1.TotalCharges) #converted into numeric (float) excluding NAN so that numerical operations can be performed
df1.TotalCharges.values


# In[14]:


df1.dtypes


# In[15]:


# identify Churn_no and Churn_yes based on tenure 
tenure_churn_no=df1[df1.Churn=='No'].tenure
tenure_churn_yes=df1[df1.Churn=='Yes'].tenure
plt.xlabel("tenure")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")
plt.hist([tenure_churn_yes,tenure_churn_no],color=['green','red'],label=['Churn=yes','Churn=No'])
plt.legend()


# In[16]:


# identify Churn_no and Churn_yes based on Monthly Charges
MC_no=df1[df1.Churn=='No'].MonthlyCharges
MC_yes=df1[df1.Churn=='Yes'].MonthlyCharges
plt.xlabel("MonthlyCharges")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")
plt.hist([MC_yes,MC_no],color=['green','red'],label=['Churn=yes','Churn=No'])
plt.legend()


# In[17]:


def print_unique_col_values(df): #making a fuinction so that I can get the unique values of each column for object datatype
    for column in df:
        if df[column].dtypes=='object':
            print(f'{column}: {df[column].unique()}') #print the unique value of each column


# In[18]:


print_unique_col_values(df1)


# In[19]:


df1.replace('No internet service','No',inplace=True)
df1.replace('No phone service','No',inplace=True)


# In[20]:


print_unique_col_values(df1)


# In[21]:


yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns:
    df1[col].replace({'Yes': 1,'No': 0},inplace=True) #replace with binary data


# In[22]:


print_unique_col_values(df1)


# In[23]:


for col in df1:
    print(f'{col}:{df1[col].unique()}')


# In[24]:


df1['gender'] = df1['gender'].map({'Female': 0, 'Male': 1})


# In[25]:


df1.gender.unique()


# In[26]:


#The get_dummies() function converts each unique value in the specified categorical columns into a new binary column, where each column represents one unique value.
#type needs to be iun int or float to do numerical calculation
df2=pd.get_dummies(data=df1,columns=['InternetService','Contract','PaymentMethod'])
df2.dtypes


# In[27]:


for col in df2:
    print(f'{col}:{df1[col].unique()}')


# In[28]:


cols_to_scale=['tenure','MonthlyCharges','TotalCharges']
from sklearn.preprocessing import MinMaxScaler #imports the MinMaxScaler class from the scikit-learn preprocessing module. The MinMaxScaler scales features to a specified range (by default, between 0 and 1).
scaler=MinMaxScaler()#Initialize MinMaxScaler
df2[cols_to_scale]=scaler.fit_transform(df2[cols_to_scale]) # The fit_transform() method calculates the minimum and maximum values of each column and scales the data accordingly. Finally, it replaces the original values in the selected columns with the scaled values.


# In[29]:


for col in df2:
    print(f'{col}:{df2[col].unique()}')


# In[30]:


X = df2.drop('Churn',axis='columns')
y = df2['Churn']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)


# In[31]:


X_train.shape


# In[32]:


X_train[:5]


# In[33]:


y_test.value_counts()


# In[34]:


X_test.shape


# In[35]:


import tensorflow as tf
from tensorflow import keras

import tensorflow as tf
from tensorflow import keras


model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)


# In[36]:


model.evaluate(X_test,y_test)


# In[37]:


yp=model.predict(X_test)
yp[:5]


# In[38]:


y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
    


# In[39]:


y_pred[:10]


# In[40]:


y_test[:10]


# In[41]:


y_test[:10]
y_pred=np.array(y_pred)


# In[42]:


from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test,y_pred))


# In[43]:


y_test.shape


# In[44]:


y_test.shape


# In[45]:


y_pred.shape


# In[46]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf

# Assuming cm is a TensorFlow tensor
sess = tf.compat.v1.Session()  # Create a TensorFlow session
with sess.as_default():
    cm_np = cm.eval()  # Evaluate the tensor to get a NumPy array

# Create heatmap
plt.figure(figsize=(10, 7))
sn.heatmap(cm_np, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()


# # Method 1: Undersample

# In[ ]:


#handling the imbalance data; y_test values 1 has 408 times, 0 has 999 times
  
#class count
count_class_0,count_class_1=df1.Churn.value_counts()

#Divide by class

df_class_0=df2[df2['Churn']== 0]
df_class_1=df2[df2['Churn']== 1]  


# In[ ]:


count_class_0,count_class_1 # 5163 number of 0's and 1869 numbers of 1


# In[ ]:


df_class_0.shape


# In[ ]:


df_class_1.shape


# In[ ]:


df_class_0.sample(count_class_1).shape  #making the same size as df_class_1 as we want to have the same numbers of 0 and 1 to handle imbalance


# In[ ]:


#we have to combine the same size of 0 and 1 together

df_class_0_under=df_class_0.sample(count_class_1)
df_test_under=pd.concat([df_class_0_under,df_class_1],axis=0) #this contains balance sample having same size of 1 and 0
df_test_under.shape


# In[ ]:


X=df_test_under.drop('Churn',axis='columns')
y=df_test_under['Churn']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=15, stratify=y) #stratify will assure same number of 0 and 1 in both X_tratin and X_test


# In[ ]:


y_train.value_counts() #same number of 0 and 1


# In[ ]:


y_test.value_counts()#same number of 0 and 1


# In[ ]:


X_train.shape


# In[ ]:


from tensorflow_addons import losses


# In[ ]:


import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix , classification_report


# In[67]:


def ANN(X_train, y_train, X_test, y_test, loss, weights):
    model = keras.Sequential([
        keras.layers.Dense(26, input_dim=26, activation='relu'),
        keras.layers.Dense(15, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    
    if weights == -1:
        model.fit(X_train, y_train, epochs=100)
    else:
        model.fit(X_train, y_train, epochs=100, class_weight = weights)
    
    print(model.evaluate(X_test, y_test))
    
    y_preds = model.predict(X_test)
    y_preds = np.round(y_preds)
    
    print("Classification Report: \n", classification_report(y_test, y_preds))
    
    return y_preds


# In[68]:


y_preds = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)

#due to balance set, the precision, recall, and F-score for 1-class has increased


# # Method2: Oversampling

# In[ ]:


count_class_0, count_class_1


# In[ ]:


df_class_1_over=df_class_1.sample(count_class_0,replace=True)# increasing the numbers in class 1 same as the size of 0
df_test_over=pd.concat([df_class_0,df_class_1_over],axis=0)#combining the daratsets
df_test_over.shape


# In[ ]:


df_class_0.shape


# In[ ]:


print(df_test_over.Churn.value_counts())


# In[ ]:


X=df_test_over.drop('Churn',axis='columns')
y=df_test_over['Churn']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=15, stratify=y) #stratify will assure same number of 0 and 1 in both X_tratin and X_test


# In[ ]:


y_train.value_counts() #checking if we have same number of 0 and 1 in y_train


# In[ ]:


y_test.value_counts()#checking if we have same number of 0 and 1 in y_test


# In[70]:


y_preds = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)


# # Method 3:SMOTE

# In[49]:


X = df2.drop('Churn',axis='columns')
y = df2['Churn']


# In[62]:


import imblearn

from imblearn.over_sampling import SMOTE

# Initialize SMOTE object
smote = SMOTE(sampling_strategy='minority')

# Fit and resample the data
X_sm, y_sm = smote.fit_resample(X, y)

# Check the value counts of the resampled data
y_sm.value_counts()


# In[63]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=15, stratify=y_sm)


# In[65]:


# Number of classes in training Data
y_train.value_counts() #checking balance


# #How SMOTE Works:
# 
# SMOTE tackles this problem by creating synthetic data points for the minority class. Here's the basic idea:
# 
# Identify the minority class: The algorithm first identifies the class with the fewest data points.
# Select a data point from the minority class: SMOTE randomly selects a data point from the minority class.
# Find its nearest neighbors: The algorithm identifies the k-nearest neighbors (similar data points) of the chosen data point within the minority class.
# Synthesize new data points: SMOTE randomly selects one of the k-nearest neighbors and creates a new data point along the line segment between the original data point and its neighbor. This new data point represents a synthetic example similar to the minority class.

# In[69]:


y_preds = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)


# # Method4: Use of Ensemble with undersampling

# In[72]:


df2.Churn.value_counts()


# In[74]:


# Regain Original features and labels
X = df2.drop('Churn',axis='columns')
y = df2['Churn']


# In[77]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)


# In[78]:


y_train.value_counts()


# In[89]:


df3=X_train.copy()
df3['Churn']=y_train

df3_class_0=df3[df3['Churn']== 0]
df3_class_1=df3[df3['Churn']== 1]  


# In[90]:


df3_class_0.shape,df3_class_1.shape


# In[91]:


df3_class_0[:1495].shape


# In[92]:


df_train=pd.concat([df3_class_0[:1495],df3_class_1],axis=0)
df_train.shape


# In[96]:


def get_train_batch(df_majority,df_minority,start,end):
    df_train=pd.concat([df_majority[start:end],df_minority],axis=0)
    X_train=df_train.drop('Churn',axis='columns')
    y_train=df_train.Churn
    return X_train,y_train


# In[98]:


X_train,y_train= get_train_batch(df3_class_0,df3_class_1,0,1495)
X_train.shape  #combines df3_class_0 and df3_class_1, but each having 1495 (starting from 0 to 1495 for X_train) 


# In[99]:


y_preds = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)


# In[100]:


X_train,y_train= get_train_batch(df3_class_0,df3_class_1,1495,1495*2) #X_train from 1495 to 2990
X_train.shape


# In[101]:


y_preds2 = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)


# In[102]:


X_train,y_train= get_train_batch(df3_class_0,df3_class_1,1495*2,4130) #X_train from o 2990 to 4130
X_train.shape


# In[103]:


y_preds3 = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)


# In[104]:


y_pred_final = y_preds.copy()
for i in range(len(y_preds)):
    n_ones = y_preds[i] + y_preds2[i] + y_preds3[i]
    if n_ones > 1:
        y_pred_final[i] = 1
    else:
        y_pred_final[i] = 0


# In[105]:


cl_rep = classification_report(y_test, y_pred_final)
print(cl_rep)


# In[ ]:




