# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 10:22:04 2017

@author: shubham
"""

'''
This file Does one hot encoding, dimensionality Reduction and Machine Learing
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.cross_validation import train_test_split
dataset = pd.read_csv('new_dataset.csv',sep = '\t',encoding = "ISO-8859-1")

dataset = dataset.join(pd.get_dummies(dataset.categories))
dataset.drop(['categories'],axis = 1,inplace = True)

dataset = dataset.join(pd.get_dummies(dataset.sellerName))
dataset.drop(['sellerName'],axis = 1,inplace=True)

dataset.set_index(['productId'],inplace= True)

dataset = dataset.join(pd.get_dummies(dataset.productBrand),lsuffix='_x',rsuffix='_r')
dataset.drop(['productBrand'],axis = 1,inplace = True)
dataset.drop(['description','color','keySpecsStr','title','imageUrlStr'],axis = 1,inplace = True)
dataset.drop(['productFamily'],axis = 1,inplace = True)

dataset.to_csv('new_dataset2.csv',sep = '\t',index = False)

dataset.inStock = dataset.inStock.astype(int)

from sklearn.cluster import KMeans

itrain,itest = train_test_split(range(dataset.shape[0]),train_size=40000, test_size=dataset.shape[0] - 40000)
xtrain = dataset[itrain].values()
xtest = dataset[itest].values()
max_score = -999
max_c = 100
clusters = [100,2000,4000,6000,10000]
for c in clusters:
    kmeans = KMeans(n_clusters = c,n_jobs = 4)
    kmeans.fit(xtrain.reshape(-1,1))
    labels = kmeans.labels_
    score = metrics.silhouette_score(xtest.reshape(-1,1), labels, metric='euclidean')
    if score > max_score:
        max_score = score
        max_c = c
        
print('Max clusters:-',max_c)

kmeans = KMeans(n_clusters = max_c,njobs = 4)
kmeans.fit(dataset)
id_labels = kmeans.labels_

labels = set(id_labels)
dict_labels = {}
count = 0
for ids in id_labels:
    dict_labels[ids].append(dataset.index[count])

