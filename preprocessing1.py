# -*- coding: utf-8 -*-

'''
This file will extract the Jeans category from the main 5Gb file. And also removes
some of the columns that are NAN, or irrelevant .
'''

import pandas as pd
import numpy as np

dataset = pd.read_csv('2oq-c1r.csv')

dataset = dataset.dropna(subset = ['categories'])     # Remove the rows that donot have any category
dataset = dataset.loc[dataset['categories'].str.endswith('>Jeans')]   #Select only the rows which have subcategory Jeans

dataset.set_index(np.arange(dataset.shape[0]),inplace=True)   #Set the index of dataset
dataset.drop(['displaySize','specificationList','sleeve','neck','idealFor'],axis = 1,inplace=True)
dataset.drop(['sizeUnit','storage'],axis = 1,inplace= True)
dataset.drop(['offers','discount','shippingCharges','deliveryTime'],axis = 1,inplace=True)
dataset.drop(['codAvailable'],axis = 1,inplace=True)
dataset.drop(['size'],axis = 1,inplace=True)

#keySpecsStr and detailedSpecsStr are always same. So, we need to take only one column.
dataset.drop(['productUrl','detailedSpecsStr'],axis = 1,inplace = True)
dataset.to_csv('new_dataset.csv',sep = '\t',index=False)