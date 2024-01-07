import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans, SpectralClustering
from psc_letter import PSC, Accuracy
import time
import torch
import argparse
import random
import warnings
import datetime
from scipy.io import arff

warnings.filterwarnings("ignore")

r = 0
print(r)
rng = np.random.RandomState(r)
torch.manual_seed(r)
random.seed(int(r))
np.random.seed(r)

parser = argparse.ArgumentParser()
parser.add_argument('-datasize', '--size', type=int, help='data size used for training')
parser.add_argument('-methods', '--methods', nargs='+', help='which method to test')
args = parser.parse_args()

data = arff.loadarff('Letter/dataset_6_letter.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].astype(str)

Class = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F':6, 'G':7, 'H':8, 'I':9, 'J':10, 'K':11, 'L':12, 'M':13, 'N':14,
            'O':15, 'P':16, 'Q':17, 'R':18, 'S':19, 'T':20, 'U':21, 'V':22, 'W':23, 'X':24, 'Y':25, 'Z':26}
df['class'] = df['class'].map(Class)
y_tmp = df['class'].values
x_data_tmp = df.drop(['class'], axis = 1).values

scaler = sklearn.preprocessing.StandardScaler().fit(x_data_tmp)
x_tmp = scaler.transform(x_data_tmp)

class Net_emb(nn.Module):
    def __init__(self):
        super(Net_emb, self).__init__()
        self.output = 26
        # Define the layers
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, self.output)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.output_layer(x)
        return x

f = open('Letter/log.txt', 'a+')
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
    # spectral_clustering = SpectralClustering(n_clusters=26, assign_labels='discretize', random_state=0)
    spectral_clustering = SpectralClustering(n_clusters=26, eigen_solver='arpack', affinity='nearest_neighbors', assign_labels='kmeans')
    # spectral_clustering = SpectralClustering(n_clusters=26)
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
    kmeans = KMeans(n_clusters=26, init='random', n_init='auto', algorithm='elkan')
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
    kmeans = KMeans(n_clusters=26, init='random', n_init='auto', algorithm='elkan', verbose=False, random_state=rng)
    psc = PSC(model=model, clustering_method=kmeans, test_splitting_rate=0, n_neighbor=10, n_components=26, random_state=rng)
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