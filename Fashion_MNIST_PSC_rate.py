import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
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
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=20, shuffle=True)

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

train_num = 60000
cut = 30000
X_train = train_data.data[:train_num]/255
X_train = X_train.unsqueeze(1) 
X_train = X_train.to(torch.float32)
X_train = autoencoder(X_train, stop=True).detach().numpy()
y_train = train_data.targets[:train_num].numpy()
print("finish data transformation")

# rate 0
print("\nrate 0")
print("1 ~ 30000")
kmeans = KMeans(n_clusters=10, init='random', n_init='auto', algorithm='elkan', random_state=0)
model = Four_layer_FNN(49, 196, 392, 196, 10)
psc = PSC(model=model, clustering_method=kmeans, test_splitting_rate=0.01, n_neighbor=10, epochs=100)

time1 = round(time.time()*1000)
PSC_index_1 = psc.fit_predict(X_train[:cut])
time2 = round(time.time()*1000)
print(f"SC time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train[:cut], y_pred=PSC_index_1)
acc.acc_report()

print("\n30001 ~ 60000")
time1 = round(time.time()*1000)
PSC_index_2 = psc.predict(X_train[:cut])
time2 = round(time.time()*1000)
print(f"SC time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train[cut:], y_pred=PSC_index_2)
acc.acc_report()

print("\n1 ~ 60000")
PSC_index = np.concatenate((PSC_index_1, PSC_index_2))
acc = Accuracy(y_true=y_train, y_pred=PSC_index)
acc.acc_report()

# rate 0.7
print("\nrate 0.7")
print("1 ~ 30000")
kmeans = KMeans(n_clusters=10, init='random', n_init='auto', algorithm='elkan', random_state=0)
model = Four_layer_FNN(49, 196, 392, 196, 10)
psc = PSC(model=model, clustering_method=kmeans, test_splitting_rate=0.7, n_neighbor=10, epochs=100)

time1 = round(time.time()*1000)
PSC_index_1 = psc.fit_predict(X_train[:cut])
time2 = round(time.time()*1000)
print(f"SC time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train[:cut], y_pred=PSC_index_1)
acc.acc_report()

print("30001 ~ 60000")
time1 = round(time.time()*1000)
PSC_index_2 = psc.predict(X_train[:cut])
time2 = round(time.time()*1000)
print(f"SC time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train[cut:], y_pred=PSC_index_2)
acc.acc_report()

print("\n1 ~ 60000")
PSC_index = np.concatenate((PSC_index_1, PSC_index_2))
acc = Accuracy(y_true=y_train, y_pred=PSC_index)
acc.acc_report()

# rate 0.5
print("\nrate 0.5")
print("1 ~ 30000")
kmeans = KMeans(n_clusters=10, init='random', n_init='auto', algorithm='elkan', random_state=0)
model = Four_layer_FNN(49, 196, 392, 196, 10)
psc = PSC(model=model, clustering_method=kmeans, test_splitting_rate=0.5, n_neighbor=10, epochs=100)

time1 = round(time.time()*1000)
PSC_index_1 = psc.fit_predict(X_train[:cut])
time2 = round(time.time()*1000)
print(f"SC time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train[:cut], y_pred=PSC_index_1)
acc.acc_report()

print("30001 ~ 60000")
time1 = round(time.time()*1000)
PSC_index_2 = psc.predict(X_train[:cut])
time2 = round(time.time()*1000)
print(f"SC time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train[cut:], y_pred=PSC_index_2)
acc.acc_report()

print("\n1 ~ 60000")
PSC_index = np.concatenate((PSC_index_1, PSC_index_2))
acc = Accuracy(y_true=y_train, y_pred=PSC_index)
acc.acc_report()

# rate 0.3
print("\nrate 0.25")
print("1 ~ 30000")
kmeans = KMeans(n_clusters=10, init='random', n_init='auto', algorithm='elkan', random_state=0)
model = Four_layer_FNN(49, 196, 392, 196, 10)
psc = PSC(model=model, clustering_method=kmeans, test_splitting_rate=0.3, n_neighbor=10, epochs=100)

time1 = round(time.time()*1000)
PSC_index_1 = psc.fit_predict(X_train[:cut])
time2 = round(time.time()*1000)
print(f"SC time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train[:cut], y_pred=PSC_index_1)
acc.acc_report()

print("30001 ~ 60000")
time1 = round(time.time()*1000)
PSC_index_2 = psc.predict(X_train[:cut])
time2 = round(time.time()*1000)
print(f"SC time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train[cut:], y_pred=PSC_index_2)
acc.acc_report()

print("\n1 ~ 60000")
PSC_index = np.concatenate((PSC_index_1, PSC_index_2))
acc = Accuracy(y_true=y_train, y_pred=PSC_index)
acc.acc_report()