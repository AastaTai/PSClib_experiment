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

train_acc_15000 = []
increment_acc_30000 = []
increment_acc_45000 = []
increment_acc_60000 = []
kmeans_acc = []
sc_acc = []

scaler = sklearn.preprocessing.StandardScaler().fit(x_tmp)
x_tmp = scaler.transform(x_tmp)

x_data_15000 = x_tmp[:15000]
y_15000 = y_tmp[:15000]

x_data_30000 = x_tmp[15000:30000]
y_30000 = y_tmp[15000:30000]


methods = args.methods

if 'kmeans' in methods:
    kmeans = KMeans(n_clusters=4, init='k-means++', n_init='auto', algorithm='elkan')
    start_time = round(time.time() * 1000)
    kmeans_index = kmeans.fit_predict(x_data_15000)
    end_time = round(time.time() * 1000)
    print("time spent:", end_time - start_time)
    acc = Accuracy(y_true=y_15000, y_pred=kmeans_index)
    kmeans_accRate, kmeans_ari, kmeans_ami = acc.acc_report()
    kmeans_acc.append(kmeans_accRate)

    kmeans_index_30000 = kmeans.fit_predict(x_data_30000)
    acc = Accuracy(y_true=y_30000, y_pred=kmeans_index_30000)
    kmeans_accRate, kmeans_ari, kmeans_ami = acc.acc_report()

    y_true_total = y_tmp[:30000]
    Kmeans_index = np.concatenate((kmeans_index, kmeans_index_30000))
    acc = Accuracy(y_true=y_true_total, y_pred=Kmeans_index)
    kmeans_accRate, kmeans_ari, kmeans_ami = acc.acc_report()