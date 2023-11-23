import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import sklearn
from sklearn import metrics
from sklearn.cluster import SpectralClustering, KMeans
from psc_covertype import PSC, Accuracy
import time
import datetime
import torch
import random
import argparse
import warnings

warnings.filterwarnings("ignore")

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('-datasize', '--size', type=int, help='data size used for training')
parser.add_argument('-methods', '--methods', nargs='+', help='which method to test')
args = parser.parse_args()

class Net_emb(nn.Module):
    def __init__(self) -> None:
        super(Net_emb, self).__init__()
        self.fc1 = nn.Linear(54, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.output_layer(x)
        return x


df = pd.read_csv("covertype.csv")
y_tmp = df['class'].values
x_tmp = df.drop(columns=['class']).values
# x_tmp = df.drop(columns=['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'class',
#                          'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',
#                          'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
#                          'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
                        #  'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']).values

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
    spectral_clustering = SpectralClustering(n_clusters=3, assign_labels='discretize', random_state=0)
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
    kmeans = KMeans(n_clusters=3, init='random', n_init='auto', algorithm='elkan')
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
    psc = PSC(model=model, clustering_method=kmeans, test_splitting_rate=0, n_neighbor=50, epochs=50)
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