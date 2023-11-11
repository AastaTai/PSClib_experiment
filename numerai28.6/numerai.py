import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans
from numerai_psc import PSC, Accuracy
import time
import torch
import random
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

df = pd.read_csv("numerai28.6/phpg2t68G.csv")
y = df['attribute_21'].values
print(y)
x_data = df.drop(columns=['attribute_21']).values

print(y[:100])

scaler = sklearn.preprocessing.StandardScaler().fit(x_data)
x = scaler.transform(x_data)

print(x)
print(f"x shape: {x.shape}")

class Net_emb(nn.Module):
    def __init__(self, out) -> None:
        super(Net_emb, self).__init__()
        self.output = out
        self.fc1 = nn.Linear(21, 32)
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

model = Net_emb(out=2)
kmeans = KMeans(n_clusters=2, init='random', n_init='auto', algorithm='elkan')
psc = PSC(model=model, clustering_method=kmeans, test_splitting_rate=0, n_neighbor=2)

print("------psc section------")
time_start = round(time.time() * 1000)
x_predict_psc = psc.fit_predict(x)
time_end = round(time.time() * 1000)
print(f"psc time spent: {time_end - time_start} milliseconds")

acc = Accuracy(y_true=y, y_pred=x_predict_psc)
acc.acc_report()
print("----psc section end---\n")


print("-----kmeans section-----")
time_start = round(time.time() * 1000)
x_predict_kmeans = kmeans.fit_predict(x)
time_end = round(time.time() * 1000)
print(f"psc time spent: {time_end - time_start} milliseconds")

acc = Accuracy(y_true=y, y_pred=x_predict_kmeans)
acc.acc_report()
print("---kmeans section end---\n")