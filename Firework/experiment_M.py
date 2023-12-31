import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import sklearn
from sklearn import metrics
from sklearn.cluster import SpectralClustering, KMeans
from psc import PSC, Accuracy
import time
import datetime
import torch
import random
import argparse
import warnings

# python Firework\experiment_M.py --methods psc sc --size 15000

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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(11, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 4)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.output_layer(x)
        return x


df = pd.read_csv('firework.csv')
action = {'allow': 1, 'deny': 2, 'drop': 3, 'reset-both': 4}
df['Action'] = df['Action'].map(action)
y_tmp = df['Action'].values
x_tmp = df.drop(['Action'], axis = 1).values

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
    # spectral_clustering = SpectralClustering(n_clusters=4, assign_labels='discretize', random_state=0)
    spectral_clustering = SpectralClustering(n_clusters=4, eigen_solver='arpack', affinity='nearest_neighbors', assign_labels='kmeans', random_state=rng)
    start_time = round(time.time()*1000)
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

#--------Parametric Spectral Clustering--------
if 'psc' in methods:
    model = Net()
    kmeans = KMeans(n_clusters=4, init='random', n_init='auto', algorithm='elkan', random_state=rng)
    psc = PSC(model=model, clustering_method=kmeans, test_splitting_rate=0, n_components=4, n_neighbor=4, batch_size_data=args.size, random_state=rng)
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

#--------PSC predict--------
    x_ = x_tmp[args.size:30000]
    start_time = round(time.time() * 1000)
    psc_index_30000 = psc.predict(x_)
    end_time = round(time.time() * 1000)
    print("time spent:", end_time - start_time)
    acc = Accuracy(y_true=y_tmp[args.size:30000], y_pred=psc_index_30000)
    psc_accRate, psc_ari, psc_ami = acc.acc_report()
    f.write("----------PSC predict 15000~30000----------\n")
    f.write("acc rate: "+str(psc_accRate)+'\n')
    f.write("ari: "+str(psc_ari)+'\n')
    f.write("ami: "+str(psc_ami)+'\n')
    f.write("time spent: " + str(end_time - start_time) + '\n\n\n')
    PSC_index = np.concatenate((psc_index, psc_index_30000))
    acc = Accuracy(y_true=y_tmp[:30000], y_pred=PSC_index)
    psc_accRate, psc_ari, psc_ami = acc.acc_report()
    f.write("----------PSC predict 0~30000----------\n")
    f.write("acc rate: "+str(psc_accRate)+'\n')
    f.write("ari: "+str(psc_ari)+'\n')
    f.write("ami: "+str(psc_ami)+'\n')
    f.write("time spent: " + str(end_time - start_time) + '\n\n\n')

    x_ = x_tmp[args.size:45000]
    start_time = round(time.time() * 1000)
    psc_index_45000 = psc.predict(x_)
    end_time = round(time.time() * 1000)
    print("time spent:", end_time - start_time)
    acc = Accuracy(y_true=y_tmp[args.size:45000], y_pred=psc_index_45000)
    psc_accRate, psc_ari, psc_ami = acc.acc_report()
    f.write("----------PSC predict 15000~45000----------\n")
    f.write("acc rate: "+str(psc_accRate)+'\n')
    f.write("ari: "+str(psc_ari)+'\n')
    f.write("ami: "+str(psc_ami)+'\n')
    f.write("time spent: " + str(end_time - start_time) + '\n\n\n')
    PSC_index = np.concatenate((psc_index, psc_index_45000))
    acc = Accuracy(y_true=y_tmp[:45000], y_pred=PSC_index)
    psc_accRate, psc_ari, psc_ami = acc.acc_report()
    f.write("----------PSC predict 0~45000----------\n")
    f.write("acc rate: "+str(psc_accRate)+'\n')
    f.write("ari: "+str(psc_ari)+'\n')
    f.write("ami: "+str(psc_ami)+'\n')
    f.write("time spent: " + str(end_time - start_time) + '\n\n\n')


    x_ = x_tmp[args.size:60000]
    start_time = round(time.time() * 1000)
    psc_index_60000 = psc.predict(x_)
    end_time = round(time.time() * 1000)
    print("time spent:", end_time - start_time)
    acc = Accuracy(y_true=y_tmp[args.size:60000], y_pred=psc_index_60000)
    psc_accRate, psc_ari, psc_ami = acc.acc_report()
    f.write("----------PSC predict 15000~60000----------\n")
    f.write("acc rate: "+str(psc_accRate)+'\n')
    f.write("ari: "+str(psc_ari)+'\n')
    f.write("ami: "+str(psc_ami)+'\n')
    f.write("time spent: " + str(end_time - start_time) + '\n\n\n')
    PSC_index = np.concatenate((psc_index, psc_index_60000))
    acc = Accuracy(y_true=y_tmp[:60000], y_pred=PSC_index)
    psc_accRate, psc_ari, psc_ami = acc.acc_report()
    f.write("----------PSC predict 0~60000----------\n")
    f.write("acc rate: "+str(psc_accRate)+'\n')
    f.write("ari: "+str(psc_ari)+'\n')
    f.write("ami: "+str(psc_ami)+'\n')
    f.write("time spent: " + str(end_time - start_time) + '\n\n\n')

f.close()

