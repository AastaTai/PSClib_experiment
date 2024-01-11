import pandas as pd
import torch.nn as nn
import sklearn
from sklearn.cluster import SpectralClustering, KMeans
from psc import PSC, Accuracy
import time
import datetime
import argparse
import warnings

# python pendigits/experiments.py --methods kmeans psc --size ${data}

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-datasize", "--size", type=int, help="data size used for training")
parser.add_argument("-methods", "--methods", nargs="+", help="which method to test")
args = parser.parse_args()


class Net_emb(nn.Module):
    def __init__(self) -> None:
        super(Net_emb, self).__init__()
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.output_layer(x)
        return x


df = pd.read_csv("pendigits/dataset_32_pendigits.csv")
y_tmp = df["class"].values
x_tmp = df.drop(columns=["id", "class"]).values

f = open("log.txt", "a+")
now = str(datetime.datetime.now())
f.write("======" + now + "======\n")
if args.size == -1:
    f.write("input data size: all\n")
    x_data = x_tmp
    y = y_tmp
else:
    f.write("input data size: " + str(args.size) + "\n")
    x_data = x_tmp[: args.size]
    y = y_tmp[: args.size]

scaler = sklearn.preprocessing.StandardScaler().fit(x_data)
x = scaler.transform(x_data)
methods = args.methods

data = pd.read_csv("pendigits/Pendigits.csv")

for i in range(10):
    # --------Spectral Clustering--------
    if "sc" in methods:
        spectral_clustering = SpectralClustering(
            n_clusters=10,
            eigen_solver="arpack",
            affinity="nearest_neighbors",
            assign_labels="kmeans",
        )
        start_time = round(time.time() * 1000)
        sc_index = spectral_clustering.fit_predict(x)
        end_time = round(time.time() * 1000)
        print("time spent:", end_time - start_time)
        acc = Accuracy(y_true=y, y_pred=sc_index)
        sc_accRate, sc_ari, sc_ami = acc.acc_report()
        f.write("---SpectralClustering---\n")
        f.write("acc rate: " + str(sc_accRate) + "\n")
        f.write("ari: " + str(sc_ari) + "\n")
        f.write("ami: " + str(sc_ami) + "\n")
        f.write("time spent: " + str(end_time - start_time) + "\n\n")

    # --------kmeans--------
    if "kmeans" in methods:
        kmeans = KMeans(n_clusters=10, init="random", n_init="auto", algorithm="elkan")
        start_time = round(time.time() * 1000)
        kmeans_index = kmeans.fit_predict(x)
        end_time = round(time.time() * 1000)
        print("time spent:", end_time - start_time)
        acc = Accuracy(y_true=y, y_pred=kmeans_index)
        kmeans_accRate, kmeans_ari, kmeans_ami = acc.acc_report()
        f.write("---------Kmeans---------\n")
        f.write("acc rate: " + str(kmeans_accRate) + "\n")
        f.write("ari: " + str(kmeans_ari) + "\n")
        f.write("ami: " + str(kmeans_ami) + "\n")
        f.write("time spent: " + str(end_time - start_time) + "\n\n")

    # --------Parametric Spectral Clustering--------
    if "psc" in methods:
        model = Net_emb()
        kmeans = KMeans(n_clusters=10, init="random", n_init="auto", algorithm="elkan")
        psc = PSC(
            model=model,
            clustering_method=kmeans,
            test_splitting_rate=0,
            n_components=10,
            n_neighbor=10,
            batch_size_data=args.size,
        )
        start_time = round(time.time() * 1000)
        psc_index = psc.fit_predict(x)
        end_time = round(time.time() * 1000)
        print("time spent:", end_time - start_time)
        acc = Accuracy(y_true=y, y_pred=psc_index)
        psc_accRate, psc_ari, psc_ami = acc.acc_report()
        f.write("----------PSC----------\n")
        f.write("acc rate: " + str(psc_accRate) + "\n")
        f.write("ari: " + str(psc_ari) + "\n")
        f.write("ami: " + str(psc_ami) + "\n")
        f.write("time spent: " + str(end_time - start_time) + "\n\n\n")
f.close()
