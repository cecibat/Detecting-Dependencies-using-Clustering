# Detecting-Dependencies-using-Clustering
This project concerns software testing with respect to test cases. It is aimed to detect dependencies between test cases using clustering algorithms.

A large number of test cases (1748) was provided; each test cases was represented by a 64-dimensions vector. For instance, these vectors were produced using a Doc2Vec algorithm. It must be noted that, due to large dimensionality, dimensionality reduction methods were tested (PCA specificlly).

A Ground Truth csv file describing the true dependencies among test cases was provided and later suitably edited accordingly to assumption and easy manipulation. Independent test cases (i.e. independent from every other test case) were not listed in this file. 

The goal is to detect these dependencies clustering together the dependent test cases; in other words, test cases belonging to different clusters are assumed to be independent.

The approach implied testing mainly two different clustering unsupervised algorithms: K-Means and HDBSCAN - using and passing different parameters. The performance of clustering algorithms' was evaluated using an evaluation metric (TruePositive, TrueNegative, FalsePositive & FalseNegative) and three specific indexes: Recall , Precision and F-Measure.

The software used was Python - specifically sklearn, numpy and hdbscan.
