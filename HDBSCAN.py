import time
start_time = time.time()
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import scipy

df = pd.read_csv('vectors.csv', sep=',', header = None)
x = np.array(df.values)
a = np.delete(x, 0,1)
"Uncomment if you want to do PCA"
#pca = PCA(0.8)
#a = pca.fit_transform(a)

import hdbscan

"Changing the metric means changing the kind of distance. The last argument means that we want not too much density!"
clusterer = hdbscan.HDBSCAN(metric='euclidean',min_cluster_size=3,cluster_selection_method='eom',gen_min_span_tree=True, p=4, alpha=0.75)
clusterer.fit(a)
print("Metric", clusterer.metric)
print("Min size", clusterer.min_cluster_size)
print("Sel method", clusterer.cluster_selection_method)
print("Alpha", clusterer.alpha)
print("Number of components",len(a[0]))

"Column vector of labels for each test case"
clusterer.labels_


"Total amount of clusters"
print("Number of clusters", clusterer.labels_.max())


"Just sorting the the labels"
lab = [];
for elem in clusterer.labels_:
    if elem not in lab:
        lab.append(elem)
sorted(lab)


"Describing results in a dictionary"   
mydict = {i: np.where(clusterer.labels_ == i)[0] for i in range(clusterer.labels_.max())}
dictlist = []
for key, value in mydict.items():
    temp = [key,value]
    dictlist.append(temp)




"Describing results in a matrix such that each row corresponds to a TC; the first element is the TC while the second is the cluster it belongs to"
res = np.empty(shape=(1748,2), dtype = int)
for i in range(len(clusterer.labels_)):
    res[i][0] = i
    res[i][1] = clusterer.labels_[i]



"This matrix is such that counts how many element there are in each cluster"
count = np.zeros(shape = (clusterer.labels_.max()+1,2), dtype = int)
for char in clusterer.labels_:
    count[int(char)][0] = char;
    count[int(char)][1] = count[int(char)][1] + 1;


"Formatted GT"
with open('Review.csv') as f:
    reader = csv.reader(f)
    df = list(reader)


TP = 0;
FN = 0;
nn = 0;
for i in range(0,1748):
    for line in df:
        if str(i) in line:
            for elem in line:
                if elem!=str(i) and int(elem) > i:
                    nn+=1;
                    if res[i][1] == res[int(elem)][1] and res[i][1]!= -1 and res[int(elem)][1] != -1:
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
elapsed_time = time.time()-start_time
print(elapsed_time)

