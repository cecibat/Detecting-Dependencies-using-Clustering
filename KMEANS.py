import time
start_time = time.time()
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import colorsys
import random
import os
import csv
import scipy

from sklearn import datasets

df = pd.read_csv('vectors.csv', sep=',', header = None)


x = np.array(df.values)
a = np.delete(x, 0,1)
#pca = PCA(0.8)
#a = pca.fit_transform(a)
print("Number of components", len(a[0]))

kmeans = KMeans(n_clusters=2000,random_state=None,n_init=10).fit(a)
labels = kmeans.labels_
print("Number of clusters", kmeans.n_clusters)
mydict = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

# Transform this dictionary into list (if you need a list as result)
dictlist = []
for key, value in mydict.items():
    temp = [key,value]
    dictlist.append(temp)

"Just sorting the the labels"
lab = [];
for elem in labels:
    if elem not in lab:
        lab.append(elem)
sorted(lab)

"Describing results in a matrix such that each row corresponds to a TC; the first element is the TC while the second is the cluster it belongs to"
res = np.empty(shape=(1748,2), dtype = int)
for i in range(kmeans.n_clusters+1):
    res[i][0] = i
    res[i][1] = labels[i]

"This matrix is such that counts how many element there are in each cluster"
count = np.zeros(shape = (kmeans.n_clusters+1,2), dtype = int)
for char in labels:
    count[int(char)][0] = char;
    count[int(char)][1] = count[int(char)][1] + 1;

"Formatted GT"
with open('Review.csv') as f:
    reader = csv.reader(f)
    df = list(reader)


TP = 0;
FN = 0;
for i in range(0,1748):
    for line in df:
        if str(i) in line:
            for elem in line:
                if elem!=str(i) and int(elem) > i:
                    if res[i][1] == res[int(elem)][1]:
                        TP = TP + 1
                    else:
                        FN = FN + 1
                    

print("TP", TP)
print("FN", FN)
recall = TP / (FN + TP)


"This is a list with all independent tcs in GT"
tc = [];
for i in range(0,1748):
     tc.append(str(i)) 

for i in range(0,1747):
    for line in df:
        if str(i) in line:
           tc.remove(str(i))


FP = 0

for i in range(0,1748):
    for j in range(0,1748):
        if j > i:
            if res[i][1] == res[j][1] and res[i][1]!= -1 and res[j][1] != -1:
                for line in df:
                    if str(i) in line and str(j) not in line:
                        FP = FP + 1

for i in range(0,1748):
    for j in range(0,1748):
        if j > i:
            if res[i][1] == res[j][1] and res[i][1]!= -1 and res[j][1] != -1:
                if str(i) in tc and str(j) in tc:
                        FP = FP + 1


precision = TP / (TP + FP)
print("FP", FP)
print("Recall", recall)
print("Precision", precision)
print("F param", 2*precision*recall/(recall+precision))
elapsed_time = time.time() - start_time
print(elapsed_time)
