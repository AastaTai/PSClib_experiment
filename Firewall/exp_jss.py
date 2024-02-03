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

# python Firewall\exp_jss.py --methods psc sc --size 15000

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


df = pd.read_csv('Firewall/firewall.csv')
action = {'allow': 1, 'deny': 2, 'drop': 3, 'reset-both': 4}
df['Action'] = df['Action'].map(action)
y_tmp = df['Action'].values
x_tmp = df.drop(['Action'], axis = 1).values

f = open('Firewall/log.txt', 'a+')
now = str(datetime.datetime.now())
f.write("======"+ now+ '======\n')

acc_15000 = []
time_15000 = []
ari_15000 = []
ami_15000 = []

acc_30000 = []
time_30000 = []
ari_30000 = []
ami_30000 = []
acc_15001_30000 = []
time_15001_30000 = []
ari_15001_30000 = []
ami_15001_30000 = []

acc_45000 = []
time_45000 = []
ari_45000 = []
ami_45000 = []
acc_15001_45000 = []
time_15001_45000 = []
ari_15001_45000 = []
ami_15001_45000 = []

acc_15001_60000 = []
time_15001_60000 = []
ari_15001_60000 = []
ami_15001_60000 = []
total_acc = []
total_time = []
total_ari = []
total_ami = []

for i in range(10):
    scaler = sklearn.preprocessing.StandardScaler().fit(x_tmp)
    x_data = scaler.transform(x_tmp)

    x_data = x_tmp
    print("end of preprocessing")

    if args.size == -1:
        f.write("input data size: all\n")
        x = x_data
        y = y_tmp
    else:
        f.write("input data size: " + str(args.size) + '\n')
        x = x_data[:args.size]
        y = y_tmp[:args.size]
    methods = args.methods

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

    if 'kmeans' in methods:
        kmeans = KMeans(n_clusters=4, init='k-means++', n_init='auto', algorithm='elkan')
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

    #--------Parametric Spectral Clustering--------
    if 'psc' in methods:
        model = Net()
        kmeans = KMeans(n_clusters=4, init='random', n_init='auto', algorithm='elkan')
        psc = PSC(model=model, clustering_method=kmeans, sampling_ratio=0, n_components=4, n_neighbor=4, batch_size_data=args.size)
        start_time = round(time.time() * 1000)
        # psc_index = psc.fit_predict(x)
        psc.fit(x)
        psc_index = psc.predict(x)
        end_time = round(time.time() * 1000)
        print("time spent:", end_time - start_time)
        acc = Accuracy(y_true=y, y_pred=psc_index)
        psc_accRate, psc_ari, psc_ami = acc.acc_report()
        pscBase_accRate = psc_accRate
        pscBase_ari = psc_ari
        pscBase_ami = psc_ami
        acc_15000.append(psc_accRate)
        ari_15000.append(psc_ari)
        ami_15000.append(psc_ami)
        time_15000.append(end_time - start_time)
        f.write("----------PSC----------\n")
        f.write("acc rate: "+str(psc_accRate)+'\n')
        f.write("ari: "+str(psc_ari)+'\n')
        f.write("ami: "+str(psc_ami)+'\n')
        f.write("time spent: " + str(end_time - start_time) + '\n\n\n')
        time_train = end_time - start_time

    #--------PSC predict--------
        x_ = x_data[args.size:30000]
        start_time = round(time.time() * 1000)
        psc_index_30000 = psc.predict(x_)
        end_time = round(time.time() * 1000)
        print("time spent:", end_time - start_time)
        acc = Accuracy(y_true=y_tmp[args.size:30000], y_pred=psc_index_30000)
        psc_accRate, psc_ari, psc_ami = acc.acc_report()
        acc_15001_30000.append(psc_accRate)
        ari_15001_30000.append(psc_ari)
        ami_15001_30000.append(psc_ami)
        time_15001_30000.append(end_time - start_time)
        f.write("----------PSC predict 15000~30000----------\n")
        f.write("acc rate: "+str(psc_accRate)+'\n')
        f.write("ari: "+str(psc_ari)+'\n')
        f.write("ami: "+str(psc_ami)+'\n')
        f.write("time spent: " + str(end_time - start_time) + '\n\n\n')

        rate = 0.5
        psc_accRate = pscBase_accRate*(1-rate) + psc_accRate*rate
        psc_ari = pscBase_ari*(1-rate) + psc_ari*rate
        psc_ami = pscBase_ami*(1-rate) + psc_ami*rate
        acc_30000.append(psc_accRate)
        ari_30000.append(psc_ari)
        ami_30000.append(psc_ami)
        time_30000.append(end_time - start_time + time_train)
        f.write("----------PSC predict 0~30000----------\n")
        f.write("acc rate: "+str(psc_accRate)+'\n')
        f.write("ari: "+str(psc_ari)+'\n')
        f.write("ami: "+str(psc_ami)+'\n')
        f.write("time spent: " + str(end_time - start_time + time_train) + '\n\n\n')

        x_ = x_data[args.size:45000]
        start_time = round(time.time() * 1000)
        psc_index_45000 = psc.predict(x_)
        end_time = round(time.time() * 1000)
        print("time spent:", end_time - start_time)
        acc = Accuracy(y_true=y_tmp[args.size:45000], y_pred=psc_index_45000)
        psc_accRate, psc_ari, psc_ami = acc.acc_report()
        acc_15001_45000.append(psc_accRate)
        ari_15001_45000.append(psc_ari)
        ami_15001_45000.append(psc_ami)
        time_15001_45000.append(end_time - start_time)
        f.write("----------PSC predict 15000~45000----------\n")
        f.write("acc rate: "+str(psc_accRate)+'\n')
        f.write("ari: "+str(psc_ari)+'\n')
        f.write("ami: "+str(psc_ami)+'\n')
        f.write("time spent: " + str(end_time - start_time) + '\n\n\n')
        
        rate = 2 / 3
        psc_accRate = pscBase_accRate*(1-rate) + psc_accRate*rate
        psc_ari = pscBase_ari*(1-rate) + psc_ari*rate
        psc_ami = pscBase_ami*(1-rate) + psc_ami*rate
        acc_45000.append(psc_accRate)
        ari_45000.append(psc_ari)
        ami_45000.append(psc_ami)
        time_45000.append(end_time - start_time + time_train)
        f.write("----------PSC predict 0~45000----------\n")
        f.write("acc rate: "+str(psc_accRate)+'\n')
        f.write("ari: "+str(psc_ari)+'\n')
        f.write("ami: "+str(psc_ami)+'\n')
        f.write("time spent: " + str(end_time - start_time + time_train) + '\n\n\n')

        x_ = x_data[args.size:60000]
        start_time = round(time.time() * 1000)
        psc_index_60000 = psc.predict(x_)
        end_time = round(time.time() * 1000)
        print("time spent:", end_time - start_time)
        acc = Accuracy(y_true=y_tmp[args.size:60000], y_pred=psc_index_60000)
        psc_accRate, psc_ari, psc_ami = acc.acc_report()
        acc_15001_60000.append(psc_accRate)
        ari_15001_60000.append(psc_ari)
        ami_15001_60000.append(psc_ami)
        time_15001_60000.append(end_time - start_time)
        f.write("----------PSC predict 15000~60000----------\n")
        f.write("acc rate: "+str(psc_accRate)+'\n')
        f.write("ari: "+str(psc_ari)+'\n')
        f.write("ami: "+str(psc_ami)+'\n')
        f.write("time spent: " + str(end_time - start_time) + '\n\n\n')
        
        rate = 0.75
        psc_accRate = pscBase_accRate*(1-rate) + psc_accRate*rate
        psc_ari = pscBase_ari*(1-rate) + psc_ari*rate
        psc_ami = pscBase_ami*(1-rate) + psc_ami*rate
        total_acc.append(psc_accRate)
        total_ari.append(psc_ari)
        total_ami.append(psc_ami)
        total_time.append(end_time - start_time + time_train)
        f.write("----------PSC predict 0~60000----------\n")
        f.write("acc rate: "+str(psc_accRate)+'\n')
        f.write("ari: "+str(psc_ari)+'\n')
        f.write("ami: "+str(psc_ami)+'\n')
        f.write("time spent: " + str(end_time - start_time + time_train) + '\n\n\n')

if 'psc' in methods:
    ari_mean, ari_std = np.mean(ari_15000), np.std(ari_15000)
    ami_mean, ami_std = np.mean(ami_15000), np.std(ami_15000)
    acc_mean, acc_std = np.mean(acc_15000), np.std(acc_15000)
    time_mean, time_std = np.mean(time_15000), np.std(time_15000)
    f.write("==============report psc[0:15000]==============\n")
    f.write("|acc: " + str(acc_mean) + "±" + str(acc_std) + '|\n')
    f.write("|ari: " + str(ari_mean) + "±" + str(ari_std) + '|\n')
    f.write("|ami: " + str(ami_mean) + "±" + str(ami_std) + '|\n')
    f.write("|time: " + str(time_mean) + "±" + str(time_std) + '|\n')
    f.write("=====================================\n\n")
    print("=========report psc[0:15000]=========")
    print("acc:", acc_mean, "±", acc_std)
    print("ari:", ari_mean, "±", ari_std)
    print("ami:", ami_mean, "±", ami_std)
    print("time:", time_mean, "±", time_std)
    print("===========================\n\n\n")

    ari_mean, ari_std = np.mean(ari_15001_30000), np.std(ari_15001_30000)
    ami_mean, ami_std = np.mean(ami_15001_30000), np.std(ami_15001_30000)
    acc_mean, acc_std = np.mean(acc_15001_30000), np.std(acc_15001_30000)
    time_mean, time_std = np.mean(time_15001_30000), np.std(time_15001_30000)
    f.write("==============report psc[15001:30000]==============\n")
    f.write("|acc: " + str(acc_mean) + "±" + str(acc_std) + '|\n')
    f.write("|ari: " + str(ari_mean) + "±" + str(ari_std) + '|\n')
    f.write("|ami: " + str(ami_mean) + "±" + str(ami_std) + '|\n')
    f.write("|time: " + str(time_mean) + "±" + str(time_std) + '|\n')
    f.write("=====================================\n\n")
    print("=========report psc[15001:30000]=========")
    print("acc:", acc_mean, "±", acc_std)
    print("ari:", ari_mean, "±", ari_std)
    print("ami:", ami_mean, "±", ami_std)
    print("time:", time_mean, "±", time_std)
    print("===========================\n\n\n")

    ari_mean, ari_std = np.mean(ari_30000), np.std(ari_30000)
    ami_mean, ami_std = np.mean(ami_30000), np.std(ami_30000)
    acc_mean, acc_std = np.mean(acc_30000), np.std(acc_30000)
    time_mean, time_std = np.mean(time_30000), np.std(time_30000)
    f.write("==============report psc[0:30000]==============\n")
    f.write("|acc: " + str(acc_mean) + "±" + str(acc_std) + '|\n')
    f.write("|ari: " + str(ari_mean) + "±" + str(ari_std) + '|\n')
    f.write("|ami: " + str(ami_mean) + "±" + str(ami_std) + '|\n')
    f.write("|time: " + str(time_mean) + "±" + str(time_std) + '|\n')
    f.write("=====================================\n\n")
    print("=========report psc[0:30000]=========")
    print("acc:", acc_mean, "±", acc_std)
    print("ari:", ari_mean, "±", ari_std)
    print("ami:", ami_mean, "±", ami_std)
    print("time:", time_mean, "±", time_std)
    print("===========================\n\n\n")

    ari_mean, ari_std = np.mean(ari_15001_45000), np.std(ari_15001_45000)
    ami_mean, ami_std = np.mean(ami_15001_45000), np.std(ami_15001_45000)
    acc_mean, acc_std = np.mean(acc_15001_45000), np.std(acc_15001_45000)
    time_mean, time_std = np.mean(time_15001_45000), np.std(time_15001_45000)
    f.write("==============report psc[15001:45000]==============\n")
    f.write("|acc: " + str(acc_mean) + "±" + str(acc_std) + '|\n')
    f.write("|ari: " + str(ari_mean) + "±" + str(ari_std) + '|\n')
    f.write("|ami: " + str(ami_mean) + "±" + str(ami_std) + '|\n')
    f.write("|time: " + str(time_mean) + "±" + str(time_std) + '|\n')
    f.write("=====================================\n\n")
    print("=========report psc[15001:45000]=========")
    print("acc:", acc_mean, "±", acc_std)
    print("ari:", ari_mean, "±", ari_std)
    print("ami:", ami_mean, "±", ami_std)
    print("time:", time_mean, "±", time_std)
    print("===========================\n\n\n")

    ari_mean, ari_std = np.mean(ari_45000), np.std(ari_45000)
    ami_mean, ami_std = np.mean(ami_45000), np.std(ami_45000)
    acc_mean, acc_std = np.mean(acc_45000), np.std(acc_45000)
    time_mean, time_std = np.mean(time_45000), np.std(time_45000)
    f.write("==============report psc[0:45000]==============\n")
    f.write("|acc: " + str(acc_mean) + "±" + str(acc_std) + '|\n')
    f.write("|ari: " + str(ari_mean) + "±" + str(ari_std) + '|\n')
    f.write("|ami: " + str(ami_mean) + "±" + str(ami_std) + '|\n')
    f.write("|time: " + str(time_mean) + "±" + str(time_std) + '|\n')
    f.write("=====================================\n\n")
    print("=========report psc[0:45000]=========")
    print("acc:", acc_mean, "±", acc_std)
    print("ari:", ari_mean, "±", ari_std)
    print("ami:", ami_mean, "±", ami_std)
    print("time:", time_mean, "±", time_std)
    print("===========================\n\n\n")

    ari_mean, ari_std = np.mean(ari_15001_60000), np.std(ari_15001_60000)
    ami_mean, ami_std = np.mean(ami_15001_60000), np.std(ami_15001_60000)
    acc_mean, acc_std = np.mean(acc_15001_60000), np.std(acc_15001_60000)
    time_mean, time_std = np.mean(time_15001_60000), np.std(time_15001_60000)
    f.write("==============report psc[15001:60000]==============\n")
    f.write("|acc: " + str(acc_mean) + "±" + str(acc_std) + '|\n')
    f.write("|ari: " + str(ari_mean) + "±" + str(ari_std) + '|\n')
    f.write("|ami: " + str(ami_mean) + "±" + str(ami_std) + '|\n')
    f.write("|time: " + str(time_mean) + "±" + str(time_std) + '|\n')
    f.write("=====================================\n\n")
    print("=========report psc[15001:60000]=========")
    print("acc:", acc_mean, "±", acc_std)
    print("ari:", ari_mean, "±", ari_std)
    print("ami:", ami_mean, "±", ami_std)
    print("time:", time_mean, "±", time_std)
    print("===========================\n\n\n")

    ari_mean, ari_std = np.mean(total_ari), np.std(total_ari)
    ami_mean, ami_std = np.mean(total_ami), np.std(total_ami)
    acc_mean, acc_std = np.mean(total_acc), np.std(total_acc)
    time_mean, time_std = np.mean(total_time), np.std(total_time)
    f.write("==============report psc[0:60000]==============\n")
    f.write("|acc: " + str(acc_mean) + "±" + str(acc_std) + '|\n')
    f.write("|ari: " + str(ari_mean) + "±" + str(ari_std) + '|\n')
    f.write("|ami: " + str(ami_mean) + "±" + str(ami_std) + '|\n')
    f.write("|time: " + str(time_mean) + "±" + str(time_std) + '|\n')
    f.write("=====================================\n\n")
    print("=========report psc[0:60000]=========")
    print("acc:", acc_mean, "±", acc_std)
    print("ari:", ari_mean, "±", ari_std)
    print("ami:", ami_mean, "±", ami_std)
    print("time:", time_mean, "±", time_std)
    print("===========================\n\n\n")

else:
    ari_mean, ari_std = np.mean(total_ari), np.std(total_ari)
    ami_mean, ami_std = np.mean(total_ami), np.std(total_ami)
    acc_mean, acc_std = np.mean(total_acc), np.std(total_acc)
    time_mean, time_std = np.mean(total_time), np.std(total_time)
    f.write("==============report==============\n")
    f.write("|acc: " + str(acc_mean) + "±" + str(acc_std) + '|\n')
    f.write("|ari: " + str(ari_mean) + "±" + str(ari_std) + '|\n')
    f.write("|ami: " + str(ami_mean) + "±" + str(ami_std) + '|\n')
    f.write("|time: " + str(time_mean) + "±" + str(time_std) + '|\n')
    f.write("=====================================\n\n")
    print("=========report=========")
    print("acc:", acc_mean, "±", acc_std)
    print("ari:", ari_mean, "±", ari_std)
    print("ami:", ami_mean, "±", ami_std)
    print("time:", time_mean, "±", time_std)
    print("===========================\n\n\n")

f.close()
