import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans, SpectralClustering
from psc import PSC, Accuracy
import time
import torch
import random
import warnings
from scipy.io import arff
import argparse
import datetime
import os

warnings.filterwarnings("ignore")

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
os.environ["PYTHONHASHSEED"] = str(0)

data = arff.loadarff('connect-4.arff')
df = pd.DataFrame(data[0])

class Net_emb(nn.Module):
    def __init__(self):
        super(Net_emb, self).__init__()
        self.output = 3
        # Define the layers
        self.fc1 = nn.Linear(42, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, self.output)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.output_layer(x)
        return x

y_tmp = df['class'].values
x_data_tmp = df.drop(['class'], axis = 1).values

scaler = sklearn.preprocessing.StandardScaler().fit(x_data_tmp)
x_tmp = scaler.transform(x_data_tmp)

parser = argparse.ArgumentParser()
parser.add_argument('-datasize', '--size', type=int, help='data size used for training')
parser.add_argument('-methods', '--methods', nargs='+', help='which method to test')
args = parser.parse_args()

f = open('log.txt', 'a+')
now = str(datetime.datetime.now())
f.write("======"+ now+ '======\n')
if args.size == -1:
    f.write("input data size: all\n")
    x_data = x_tmp
    y = y_tmp
else:
    f.write("input data size: " + str(args.size) + '\n')
    x_data = x_tmp[:args.size]
    y = y_tmp[:args.size]

scaler = sklearn.preprocessing.StandardScaler().fit(x_data)
x = scaler.transform(x_data)
methods = args.methods

#--------Spectral Clustering--------
if 'sc' in methods:
    spectral_clustering = SpectralClustering(n_clusters=10, assign_labels='discretize', random_state=0)
    start_time = round(time.time() * 1000)
    sc_index = spectral_clustering.fit_predict(x)
    end_time = round(time.time()*1000)
    print("time spent:", end_time-start_time)
    acc = Accuracy(y_true=y, y_pred=sc_index)
    sc_accRate, sc_ari, sc_ami = acc.acc_report()
    f.write("---SpectralClustering---\n")
    f.write("acc rate: "+str(sc_accRate)+'\n')
    f.write("ari: "+str(sc_ari)+'\n')
    f.write("ami: "+str(sc_ami)+'\n')
    f.write("time spent: " + str(end_time - start_time) + '\n\n')

#--------kmeans--------
if 'kmeans' in methods:
    kmeans = KMeans(n_clusters=10, init='random', n_init='auto', algorithm='elkan')
    start_time = round(time.time() * 1000)
    kmeans_index = kmeans.fit_predict(x)
    end_time = round(time.time() * 1000)
    print("time spent:", end_time - start_time)
    acc = Accuracy(y_true=y, y_pred=kmeans_index)
    kmeans_accRate, kmeans_ari, kmeans_ami = acc.acc_report()
    f.write("---------Kmeans---------\n")
    f.write("acc rate: "+str(kmeans_accRate)+'\n')
    f.write("ari: "+str(kmeans_ari)+'\n')
    f.write("ami: "+str(kmeans_ami)+'\n')
    f.write("time spent: " + str(end_time - start_time) + '\n\n')

#--------Parametric Spectral Clustering--------
if 'psc' in methods:
    model = Net_emb()
    psc = PSC(model=model, clustering_method=kmeans, test_splitting_rate=0, n_neighbor=20, epochs=50)
    start_time = round(time.time() * 1000)
    psc_index = psc.fit_predict(x)
    end_time = round(time.time() * 1000)
    print("time spent:", end_time - start_time)
    acc = Accuracy(y_true=y, y_pred=psc_index)
    psc_accRate, psc_ari, psc_ami = acc.acc_report()
    f.write("----------PSC----------\n")
    f.write("acc rate: "+str(psc_accRate)+'\n')
    f.write("ari: "+str(psc_ari)+'\n')
    f.write("ami: "+str(psc_ami)+'\n')
    f.write("time spent: " + str(end_time - start_time) + '\n\n\n')
f.close()