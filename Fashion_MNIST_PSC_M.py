import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans, BisectingKMeans, MiniBatchKMeans, AffinityPropagation, AgglomerativeClustering, Birch
import numpy as np
import random
import time
from psc import Accuracy, PSC, Four_layer_FNN

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False),
            nn.Conv2d(8, 1, 3, 1, 1),
            nn.ReLU(inplace=False),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(8, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, stop=False):
        x = self.encoder(x)
        if stop:
            x = x.view(x.size(0), -1)
            return x
        x = self.decoder(x)
        return x


train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
# train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=20, shuffle=True)

autoencoder = Autoencoder()
# criterion = nn.MSELoss()
# optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
# for epoch in range(30):
#     running_loss = 0.0
#     for x,y in train_loader:
#         # x = x.to(device)
#         optimizer.zero_grad()
#         output = autoencoder(x)
#         loss = criterion(output, x)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         Loss = running_loss/len(train_loader)
#     print(f"Epoch {epoch+1}, Loss: {Loss:.4f}")

# torch.save(autoencoder,'autoencoder_simple.pth')
autoencoder = torch.load("autoencoder_simple.pth")

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(7*7, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 10)

    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.output_layer(x)

        return x
model = Net1()

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(49, 196)
        self.fc2 = nn.Linear(196, 392)
        self.fc3 = nn.Linear(392, 196)
        self.fc4 = nn.Linear(196, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.softmax(x)

        return x
# model = Net2()

train_num = 60000
cut_1 = 15000
cut_2 = 30000
cut_3 = 45000
X_train = train_data.data[:train_num]/255
X_train = X_train.unsqueeze(1) 
X_train = X_train.to(torch.float32)
X_train = autoencoder(X_train, stop=True).detach().numpy()
y_train = train_data.targets[:train_num].numpy()
print("finish data transformation")
# model = Four_layer_FNN(49, 196, 392, 196, 10)


# 1 ~ 15000
print("\n1 ~ 15000")
kmeans = KMeans(n_clusters=10, init='random', n_init='auto', algorithm='elkan')
psc = PSC(model=model, clustering_method=kmeans, test_splitting_rate=0, n_neighbor=10, epochs=100)

time1 = round(time.time()*1000)
SC_index_trained = psc.fit_predict(X_train[:cut_1])
time2 = round(time.time()*1000)
print(f"PSC time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train[:cut_1], y_pred=SC_index_trained)
acc.acc_report()

# 1 ~ 30000
print("\n1 ~ 30000")

time1 = round(time.time()*1000)
SC_index = psc.predict(X_train[cut_1:cut_2])
time2 = round(time.time()*1000)
print(f"PSC time spent: {time2 - time1} milliseconds")

print("predict() for 15000 datas")
acc = Accuracy(y_true=y_train[cut_1:cut_2], y_pred=SC_index)
acc.acc_report()

print("All 30000 datas")
SC_index = np.concatenate((SC_index_trained, SC_index))
acc = Accuracy(y_true=y_train[:cut_2], y_pred=SC_index)
acc.acc_report()


# 1 ~ 45000
print("\n1 ~ 45000")

time1 = round(time.time()*1000)
SC_index = psc.predict(X_train[cut_1:cut_3])
time2 = round(time.time()*1000)
print(f"PSC time spent: {time2 - time1} milliseconds")

print("predict() for 30000 datas")
acc = Accuracy(y_true=y_train[cut_1:cut_3], y_pred=SC_index)
acc.acc_report()

print("All 45000 datas")
SC_index = np.concatenate((SC_index_trained, SC_index))
acc = Accuracy(y_true=y_train[:cut_3], y_pred=SC_index)
acc.acc_report()

# 1 ~ 60000
print("\n1 ~ 60000")

time1 = round(time.time()*1000)
SC_index = psc.predict(X_train[cut_1:train_num])
time2 = round(time.time()*1000)
print(f"PSC time spent: {time2 - time1} milliseconds")

print("predict() for 45000 datas")
acc = Accuracy(y_true=y_train[cut_1:train_num], y_pred=SC_index)
acc.acc_report()

print("All 60000 datas")
SC_index = np.concatenate((SC_index_trained, SC_index))
acc = Accuracy(y_true=y_train[:train_num], y_pred=SC_index)
acc.acc_report()