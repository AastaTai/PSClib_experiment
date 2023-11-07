import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans
from pendigits_psc import PSC, Accuracy
import time
import torch
import random
import warnings
from datetime import datetime

# warnings.filterwarnings("ignore")

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

df = pd.read_csv("pendigits/dataset_32_pendigits.csv")
y = df['class'].values
print(y)
x_data = df.drop(columns=['id', 'class']).values


scaler = sklearn.preprocessing.StandardScaler().fit(x_data)
x = scaler.transform(x_data)

print(x)
print("x shape:", x.shape)

class Net_emb(nn.Module):
    def __init__(self, out) -> None:
        super(Net_emb, self).__init__()
        self.output = out
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

model = Net_emb(out=10)
kmeans = KMeans(n_clusters=10, init='random', n_init='auto', algorithm='elkan')
psc = PSC(model=model, clustering_method=kmeans, test_splitting_rate=0, n_neighbor=10, epochs=50)

# f = open('pendigits/log.txt', 'w')
# current_time = datetime.now()
# f.write("\n\n=======" + str(current_time) + "=======")

print("------psc section------")
time1 = round(time.time() * 1000)
x_predict_psc = psc.fit_predict(x)
time2 = round(time.time() * 1000)
print(f"psc time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y, y_pred=x_predict_psc)
acc.acc_report()
print("----psc section end----\n")

print("-----kmeans section-----")
time_start = round(time.time() * 1000)
x_predict_kmeans = kmeans.fit_predict(x)
time_end = round(time.time() * 1000)
print(f"kmeans time spent: {time_end - time_start} milliseconds")

acc = Accuracy(y_true=y, y_pred=x_predict_kmeans)
acc.acc_report()
print("---kmeans section end---")
