#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation

# ## Task 2 : Prediction using unsupervised ML

# ### Problem Statement : We have to predict the optimum number of clusters and represent it visually.

# ### Author : Pramita Agarwal

# ## Importing Libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets


# ## Loading the iris dataset

# In[2]:


iris = datasets.load_iris()                                             #getting first 5 rows of dataset
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_df.head() 


# ## Analysing and understanding the data

# In[3]:


iris_df.shape                        #provides the shape of data


# In[6]:


iris_df.describe()


# In[8]:


iris_df.isnull().sum()                     #checkinf for null values 


# ## Using Elbow method 

# In[14]:


# Finding the optimum number of clusters for k-means classification

x = iris_df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)                             # Plotting the results onto a line graph
plt.title('The elbow method')                            # Allowing us to observe 'The elbow'
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')                                       # Within cluster sum of squares
plt.show()


# #### From this we choose the number of clusters as "3" 

# In[10]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',               #Creating the kmeans classifier
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[11]:


y_kmeans


# ## Visualising the clusters

# In[12]:


cl =pd.Series(kmeans.labels_)
iris_df['cluster'] = cl
iris_df.iloc[:,:]


# In[13]:


# Visualising the clusters - On the first two columns
plt.figure(figsize=(9,6))
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
 s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# In[ ]:




