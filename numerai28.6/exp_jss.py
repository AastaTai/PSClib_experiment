import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans, SpectralClustering
from numerai_psc import PSC, Accuracy
import time
import torch
import random
import warnings
import argparse
import datetime

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-datasize', '--size', type=int, help='data size used for training')
parser.add_argument('-methods', '--methods', nargs='+', help='which method to test')
args = parser.parse_args()

class Net_emb(nn.Module):
    def __init__(self) -> None:
        super(Net_emb, self).__init__()
        output = 2
        self.fc1 = nn.Linear(21, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.output_layer(x)
        return x

df = pd.read_csv("phpg2t68G.csv")
y_tmp = df['attribute_21'].values
x_tmp = df.drop(columns=['attribute_21']).values

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

if 'psc' in methods:
    total_acc = []
    total_time = []
    total_ari = []
    total_ami = []
    for _ in range(10):
        model = Net_emb()
        kmeans = KMeans(n_clusters=2, init='random', n_init='auto', algorithm='elkan')
        psc = PSC(model=model, clustering_method=kmeans, test_splitting_rate=0, n_neighbor=10, n_components=2)
        start_time = round(time.time() * 1000)
        psc_index = psc.fit_predict(x)
        end_time = round(time.time() * 1000)
        print("time spent:", end_time - start_time)
        acc = Accuracy(y_true=y, y_pred=psc_index)
        psc_accRate, psc_ari, psc_ami = acc.acc_report()
        total_acc.append(psc_accRate)
        total_ari.append(psc_ari)
        total_ami.append(psc_ami)
        total_time.append(end_time-start_time)
        f.write("----------PSC----------\n")
        f.write("acc rate: "+str(psc_accRate)+'\n')
        f.write("ari: "+str(psc_ari)+'\n')
        f.write("ami: "+str(psc_ami)+'\n')
        f.write("time spent: " + str(end_time - start_time) + '\n\n\n')

    ari_mean, ari_std = np.mean(total_ari), np.std(total_ari)
    ami_mean, ami_std = np.mean(total_ami), np.std(total_ami)
    acc_mean, acc_std = np.mean(total_acc), np.std(total_acc)
    time_mean, time_std = np.mean(total_time), np.std(total_time)
    f.write("==============PSC report==============\n")
    f.write("|acc: " + str(acc_mean) + "±" + str(acc_std) + '|\n')
    f.write("|ari: " + str(ari_mean) + "±" + str(ari_std) + '|\n')
    f.write("|ami: " + str(ami_mean) + "±" + str(ami_std) + '|\n')
    f.write("|time: " + str(time_mean) + "±" + str(time_std) + '|\n')
    f.write("======================================\n\n")

    print("=========PSC report=========")
    print("acc:", acc_mean, "±", acc_std)
    print("ari:", ari_mean, "±", ari_std)
    print("ami:", ami_mean, "±", ami_std)
    print("time:", time_mean, "±", time_std)
    print("============================\n\n\n")

if 'sc' in methods:
    total_acc = []
    total_time = []
    total_ari = []
    total_ami = []
    for i in range(10):
        print(i)
        spectral_clustering = SpectralClustering(n_clusters=2, assign_labels='discretize')
        start_time = round(time.time() * 1000)
        sc_index = spectral_clustering.fit_predict(x)
        end_time = round(time.time()*1000)
        print("time spent:", end_time-start_time)
        acc = Accuracy(y_true=y, y_pred=sc_index)
        sc_accRate, sc_ari, sc_ami = acc.acc_report()
        total_acc.append(sc_accRate)
        total_ari.append(sc_ari)
        total_ami.append(sc_ami)
        total_time.append(end_time-start_time)
        f.write("---SpectralClustering---\n")
        f.write("acc rate: "+str(sc_accRate)+'\n')
        f.write("ari: "+str(sc_ari)+'\n')
        f.write("ami: "+str(sc_ami)+'\n')
        f.write("time spent: " + str(end_time - start_time) + '\n\n')

    ari_mean, ari_std = np.mean(total_ari), np.std(total_ari)
    ami_mean, ami_std = np.mean(total_ami), np.std(total_ami)
    acc_mean, acc_std = np.mean(total_acc), np.std(total_acc)
    time_mean, time_std = np.mean(total_time), np.std(total_time)
    f.write("==============SC report==============\n")
    f.write("|acc: " + str(acc_mean) + "±" + str(acc_std) + '|\n')
    f.write("|ari: " + str(ari_mean) + "±" + str(ari_std) + '|\n')
    f.write("|ami: " + str(ami_mean) + "±" + str(ami_std) + '|\n')
    f.write("|time: " + str(time_mean) + "±" + str(time_std) + '|\n')
    f.write("=====================================\n\n")

    print("=========SC report=========")
    print("acc:", acc_mean, "±", acc_std)
    print("ari:", ari_mean, "±", ari_std)
    print("ami:", ami_mean, "±", ami_std)
    print("time:", time_mean, "±", time_std)
    print("===========================\n\n\n")

if 'kmeans' in methods:
    total_acc = []
    total_time = []
    total_ari = []
    total_ami = []
    for _ in range(10):
        kmeans = KMeans(n_clusters=2, init='random', n_init='auto', algorithm='elkan')
        start_time = round(time.time() * 1000)
        kmeans_index = kmeans.fit_predict(x)
        end_time = round(time.time() * 1000)
        print("time spent:", end_time - start_time)
        acc = Accuracy(y_true=y, y_pred=kmeans_index)
        kmeans_accRate, kmeans_ari, kmeans_ami = acc.acc_report()
        total_acc.append(kmeans_accRate)
        total_ari.append(kmeans_ari)
        total_ami.append(kmeans_ami)
        total_time.append(end_time - start_time)
        f.write("---------Kmeans---------\n")
        f.write("acc rate: "+str(kmeans_accRate)+'\n')
        f.write("ari: "+str(kmeans_ari)+'\n')
        f.write("ami: "+str(kmeans_ami)+'\n')
        f.write("time spent: " + str(end_time - start_time) + '\n\n')

    ari_mean, ari_std = np.mean(total_ari), np.std(total_ari)
    ami_mean, ami_std = np.mean(total_ami), np.std(total_ami)
    acc_mean, acc_std = np.mean(total_acc), np.std(total_acc)
    time_mean, time_std = np.mean(total_time), np.std(total_time)
    f.write("==============kmeans report==============\n")
    f.write("|acc: " + str(acc_mean) + "±" + str(acc_std) + '|\n')
    f.write("|ari: " + str(ari_mean) + "±" + str(ari_std) + '|\n')
    f.write("|ami: " + str(ami_mean) + "±" + str(ami_std) + '|\n')
    f.write("|time: " + str(time_mean) + "±" + str(time_std) + '|\n')
    f.write("=========================================\n\n")

    print("=========kmeans report=========")
    print("acc:", acc_mean, "±", acc_std)
    print("ari:", ari_mean, "±", ari_std)
    print("ami:", ami_mean, "±", ami_std)
    print("time:", time_mean, "±", time_std)
    print("===============================\n\n\n")

f.close()