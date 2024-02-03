import pandas as pd
import numpy as np
import torch.nn as nn
import sklearn
from sklearn.cluster import SpectralClustering, KMeans
from psc import PSC, Accuracy
import time
import datetime
import argparse
import warnings

# python Firewall\experiment_M_SplittingRate.py --methods psc --size 15000 --rate 0

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-datasize', '--size', type=int, help='data size used for training')
parser.add_argument('-methods', '--methods', nargs='+', help='which method to test')
parser.add_argument('-rate', '--rate', type=float, help='test splitting rate')
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


df = pd.read_csv('Firewall/firewall.csv')
action = {'allow': 1, 'deny': 2, 'drop': 3, 'reset-both': 4}
df['Action'] = df['Action'].map(action)
y_tmp = df['Action'].values
x_tmp = df.drop(['Action'], axis = 1).values

f = open('Firewall/log.txt', 'a+')
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

data = pd.read_csv('Firewall/Firewall_M.csv')

first_acc = []
first_time = []
first_ari = []
first_ami = []
second_acc = []
second_time = []
second_ari = []
second_ami = []
total_acc = []
total_time = []
total_ari = []
total_ami = []

for i in range(10):
    #--------Spectral Clustering--------
    if 'sc' in methods:
        spectral_clustering = SpectralClustering(n_clusters=4, eigen_solver='arpack', affinity='nearest_neighbors', assign_labels='kmeans')
        start_time = round(time.time()*1000)
        sc_index = spectral_clustering.fit_predict(x)
        end_time = round(time.time()*1000)
        print("time spent:", end_time-start_time)
        acc = Accuracy(y_true=y, y_pred=sc_index)
        sc_accRate, sc_ari, sc_ami = acc.acc_report()
        total_acc.append(sc_accRate)
        total_ari.append(sc_ari)
        total_ami.append(sc_ami)
        total_time.append(end_time - start_time)
        f.write("---SpectralClustering---\n")
        f.write("acc rate: "+str(sc_accRate)+'\n')
        f.write("ari: "+str(sc_ari)+'\n')
        f.write("ami: "+str(sc_ami)+'\n')
        f.write("time spent: " + str(end_time - start_time) + '\n\n')
        data.loc[len(data)] = ['Spectral Clustering', args.size, sc_accRate, sc_ari, sc_ami, end_time-start_time]

    #--------Parametric Spectral Clustering--------
    if 'psc' in methods:
        model = Net()
        kmeans = KMeans(n_clusters=4, init='random', n_init='auto', algorithm='elkan')
        psc = PSC(model=model, clustering_method=kmeans, sampling_ratio=args.rate, n_components=4, n_neighbor=4, batch_size_data=args.size)
        start_time = round(time.time() * 1000)
        psc.fit(x)
        psc_index = psc.predict(x[0:15000])
        end_time = round(time.time() * 1000)
        print("time spent:", end_time - start_time)
        base_time = end_time - start_time
        acc = Accuracy(y_true=y[0:15000], y_pred=psc_index)
        pscBase_accRate, pscBase_ari, pscBase_ami = acc.acc_report()
        first_acc.append(pscBase_accRate)
        first_ari.append(pscBase_ari)
        first_ami.append(pscBase_ami)
        first_time.append(end_time - start_time)
        f.write("----------PSC----------\n")
        f.write("acc rate: "+str(pscBase_accRate)+'\n')
        f.write("ari: "+str(pscBase_ari)+'\n')
        f.write("ami: "+str(pscBase_ami)+'\n')
        f.write("time spent: " + str(end_time - start_time) + '\n\n\n')
        data.loc[len(data)] = ['PSC', args.size, pscBase_accRate, pscBase_ari, pscBase_ami, end_time-start_time]
        
        if args.size > 15000:
            #--------15000 ~ args.size--------
            start_time = round(time.time() * 1000)
            psc_index = psc.predict(x[15000:args.size])
            end_time = round(time.time() * 1000)
            print("time spent:", end_time - start_time)
            acc = Accuracy(y_true=y[15000:args.size], y_pred=psc_index)
            psc_accRate, psc_ari, psc_ami = acc.acc_report()
            second_acc.append(psc_accRate)
            second_ari.append(psc_ari)
            second_ami.append(psc_ami)
            second_time.append(end_time - start_time)
            f.write("----------PSC 15000 ~ " + str(args.size) + "----------\n")
            f.write("acc rate: "+str(psc_accRate)+'\n')
            f.write("ari: "+str(psc_ari)+'\n')
            f.write("ami: "+str(psc_ami)+'\n')
            f.write("time spent: " + str(end_time - start_time) + '\n\n\n')
            data.loc[len(data)] = ['PSC', args.size, psc_accRate, psc_ari, psc_ami, end_time-start_time]
            
            #--------0 ~ args.size--------
            psc_accRate = pscBase_accRate*(1-args.rate) + psc_accRate*args.rate
            psc_ari = pscBase_ari*(1-args.rate) + psc_ari*args.rate
            psc_ami = pscBase_ami*(1-args.rate) + psc_ami*args.rate
            total_acc.append(psc_accRate)
            total_ari.append(psc_ari)
            total_ami.append(psc_ami)
            total_time.append(end_time - start_time + base_time)
            f.write("----------PSC 0 ~ " + str(args.size) + "----------\n")
            f.write("acc rate: "+str(psc_accRate)+'\n')
            f.write("ari: "+str(psc_ari)+'\n')
            f.write("ami: "+str(psc_ami)+'\n')
            f.write("time spent: " + str(end_time - start_time + base_time) + '\n\n\n')
            data.loc[len(data)] = ['PSC predict', '0~45000', psc_accRate, psc_ari, psc_ami, end_time-start_time+base_time]

if 'sc' in methods:
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

if 'psc' in methods:
    ari_mean, ari_std = np.mean(first_ari), np.std(first_ari)
    ami_mean, ami_std = np.mean(first_ami), np.std(first_ami)
    acc_mean, acc_std = np.mean(first_acc), np.std(first_acc)
    time_mean, time_std = np.mean(first_time), np.std(first_time)
    f.write("==============PSC report==============\n")
    f.write("|acc: " + str(acc_mean) + "±" + str(acc_std) + '|\n')
    f.write("|ari: " + str(ari_mean) + "±" + str(ari_std) + '|\n')
    f.write("|ami: " + str(ami_mean) + "±" + str(ami_std) + '|\n')
    f.write("|time: " + str(time_mean) + "±" + str(time_std) + '|\n')
    f.write("=====================================\n\n")

    print("=========PSC report=========")
    print("acc:", acc_mean, "±", acc_std)
    print("ari:", ari_mean, "±", ari_std)
    print("ami:", ami_mean, "±", ami_std)
    print("time:", time_mean, "±", time_std)
    print("===========================\n\n\n")
    
    if args.size > 15000:
        #--------15000 ~ args.size--------
        ari_mean, ari_std = np.mean(second_ari), np.std(second_ari)
        ami_mean, ami_std = np.mean(second_ami), np.std(second_ami)
        acc_mean, acc_std = np.mean(second_acc), np.std(second_acc)
        time_mean, time_std = np.mean(second_time), np.std(second_time)
        f.write("==============PSC report==============\n")
        f.write("|acc: " + str(acc_mean) + "±" + str(acc_std) + '|\n')
        f.write("|ari: " + str(ari_mean) + "±" + str(ari_std) + '|\n')
        f.write("|ami: " + str(ami_mean) + "±" + str(ami_std) + '|\n')
        f.write("|time: " + str(time_mean) + "±" + str(time_std) + '|\n')
        f.write("=====================================\n\n")
        
        print("=========PSC report=========")
        print("acc:", acc_mean, "±", acc_std)
        print("ari:", ari_mean, "±", ari_std)
        print("ami:", ami_mean, "±", ami_std)
        print("time:", time_mean, "±", time_std)
        print("===========================\n\n\n")
        #--------0 ~ args.size--------
        ari_mean, ari_std = np.mean(total_ari), np.std(total_ari)
        ami_mean, ami_std = np.mean(total_ami), np.std(total_ami)
        acc_mean, acc_std = np.mean(total_acc), np.std(total_acc)
        time_mean, time_std = np.mean(total_time), np.std(total_time)
        f.write("==============PSC report==============\n")
        f.write("|acc: " + str(acc_mean) + "±" + str(acc_std) + '|\n')
        f.write("|ari: " + str(ari_mean) + "±" + str(ari_std) + '|\n')
        f.write("|ami: " + str(ami_mean) + "±" + str(ami_std) + '|\n')
        f.write("|time: " + str(time_mean) + "±" + str(time_std) + '|\n')
        f.write("=====================================\n\n")

        print("=========PSC report=========")
        print("acc:", acc_mean, "±", acc_std)
        print("ari:", ari_mean, "±", ari_std)
        print("ami:", ami_mean, "±", ami_std)
        print("time:", time_mean, "±", time_std)
        print("===========================\n\n\n")

print(data)
data.to_csv('Firewall/Firewall_M.csv', index=False)
f.close()