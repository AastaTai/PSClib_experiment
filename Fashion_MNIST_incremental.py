import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
from sklearn.cluster import KMeans, SpectralClustering
import time
from ParametricSpectralClustering import PSC, Accuracy, Four_layer_FNN

# torch.manual_seed(0)
# np.random.seed(0)
# random.seed(0)

# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 64, 7, 1, 3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, 5, 1, 2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(64, 32, 5, 1, 2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, 5, 1, 2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(32, 1, 3, 1, 1),
#         )
        
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(1, 32, 3, 1, 1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(32, 32, 4, 2, 1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(32, 64, 5, 1, 2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, 64, 4, 2, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, 1, 7, 1, 3),
#         )

#     def forward(self, x, stop=False):
#         x = self.encoder(x)
#         if stop:
#             x = x.view(x.size(0), -1)
#             return x
#         x = self.decoder(x)
#         return x

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

class Dimen_Reduct_Model(nn.Module):
    def __init__(self):
        super(Dimen_Reduct_Model, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(49, 196),
            nn.ReLU(inplace=False),
            nn.Linear(196, 392),
            nn.ReLU(inplace=False),
            nn.Linear(392, 196),
            nn.ReLU(inplace=False),
            nn.Linear(196, 10)
        )
        
    def forward(self, x):
        x = self.classifier(x)
        return x

class MyIterableDataset(IterableDataset):

    def __init__(self, data):
        self.data = data
    
    def __iter__(self):
        return self.get_stream(self.data)
    
    def get_stream(self, data):
        for i in range(len(data)):
            yield data[i]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

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
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

# iterable_dataset = MyIterableDataset(train_data)
# train_loader_it = DataLoader(iterable_dataset, batch_size=10, shuffle=True)

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
cut_1 = 15000
cut_2 = 30000
cut_3 = 45000
X_train = train_data.data[:cut_1]/255
X_train = X_train.unsqueeze(1) 
X_train = X_train.to(torch.float32)
X_train = autoencoder(X_train, stop=True).detach().numpy()
y_train = train_data.targets[:cut_1].numpy()
print("finish data transformation")


# sc = SpectralClustering(n_clusters=10, assign_labels='discretize', random_state=0)

# 0 ~ 15000 fit + predict
print("\n0 ~ 15000")
kmeans = KMeans(n_clusters=10, init="k-means++", n_init=1, max_iter=100, algorithm='elkan')
time1 = round(time.time()*1000)
Kmeans_index = kmeans.fit_predict(X_train)
time2 = round(time.time()*1000)
print(f"Kmeans time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train, y_pred=Kmeans_index)
acc.acc_report()

sc = SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity='nearest_neighbors', assign_labels='kmeans')
time1 = round(time.time()*1000)
SC_index = sc.fit_predict(X_train)
time2 = round(time.time()*1000)
print(f"SC time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train, y_pred=SC_index)
acc.acc_report()

sc = SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity='nearest_neighbors', assign_labels='kmeans')
model = Four_layer_FNN(49, 196, 392, 196, 10)
psc = PSC(model=model, clustering_method=kmeans, test_splitting_rate=0.7)
time1 = round(time.time()*1000)
PSC_index = psc.fit_predict(X_train)
time2 = round(time.time()*1000)
print(f"PSC time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train, y_pred=PSC_index)
acc.acc_report()

# 15001 ~ 30000 predict
print("\n15001 ~ 30000")
X_train = train_data.data[cut_1:cut_2]/255
X_train = X_train.unsqueeze(1) 
X_train = X_train.to(torch.float32)
X_train = autoencoder(X_train, stop=True).detach().numpy()
y_train = train_data.targets[cut_1:cut_2].numpy()

time1 = round(time.time()*1000)
Kmeans_index = kmeans.fit_predict(X_train)
time2 = round(time.time()*1000)
print(f"Kmeans time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train, y_pred=Kmeans_index)
acc.acc_report()

time1 = round(time.time()*1000)
SC_index = sc.fit_predict(X_train)
time2 = round(time.time()*1000)
print(f"SC time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train, y_pred=SC_index)
acc.acc_report()

time1 = round(time.time()*1000)
PSC_index = psc.predict(X_train)
time2 = round(time.time()*1000)
print(f"PSC time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train, y_pred=PSC_index)
acc.acc_report()

# 30001 ~ 45000 predict
print("\n30001 ~ 45000")
X_train = train_data.data[cut_2:cut_3]/255
X_train = X_train.unsqueeze(1) 
X_train = X_train.to(torch.float32)
X_train = autoencoder(X_train, stop=True).detach().numpy()
y_train = train_data.targets[cut_2:cut_3].numpy()

time1 = round(time.time()*1000)
SC_index = kmeans.fit_predict(X_train)
time2 = round(time.time()*1000)
print(f"Kmeans time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train, y_pred=Kmeans_index)
acc.acc_report()

time1 = round(time.time()*1000)
SC_index = sc.fit_predict(X_train)
time2 = round(time.time()*1000)
print(f"SC time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train, y_pred=SC_index)
acc.acc_report()

time1 = round(time.time()*1000)
PSC_index = psc.predict(X_train)
time2 = round(time.time()*1000)
print(f"PSC time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train, y_pred=PSC_index)
acc.acc_report()

# 45001 ~ 60000 predict
print("\n45001 ~ 60000")
X_train = train_data.data[cut_3:train_num]/255
X_train = X_train.unsqueeze(1) 
X_train = X_train.to(torch.float32)
X_train = autoencoder(X_train, stop=True).detach().numpy()
y_train = train_data.targets[cut_3:train_num].numpy()

time1 = round(time.time()*1000)
Kmeans_index = kmeans.fit_predict(X_train)
time2 = round(time.time()*1000)
print(f"Kmeans time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train, y_pred=Kmeans_index)
acc.acc_report()

time1 = round(time.time()*1000)
sc = SpectralClustering(n_clusters=10, assign_labels='discretize', random_state=0)
SC_index = sc.fit_predict(X_train)
time2 = round(time.time()*1000)
print(f"SC time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train, y_pred=SC_index)
acc.acc_report()

time1 = round(time.time()*1000)
PSC_index = psc.predict(X_train)
time2 = round(time.time()*1000)
print(f"PSC time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train, y_pred=PSC_index)
acc.acc_report()