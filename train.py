import os
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import DNN

from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = True

# ==================================================================================================
# ======================================== Hyper Parameters ========================================
# ==================================================================================================

topN = 1000
embedding_dim = 10

epoch_size = 100
batch_size = 500

model = DNN(vocab_size=topN + 2, embedding_dim=embedding_dim, vector_len=80).to(device)

lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)
loss = torch.nn.BCELoss().to(device)

dataPath = "./dataSet"
train_x = np.load(os.path.join(dataPath, "train_x_for_top_%d.npy" % (topN)))
train_y = np.load(os.path.join(dataPath, "train_y_for_top_%d.npy" % (topN)))
test_x = np.load(os.path.join(dataPath, "test_x_for_top_%d.npy" % (topN)))
test_y = np.load(os.path.join(dataPath, "test_y_for_top_%d.npy" % (topN)))

# ==================================================================================================
# ======================================== print train info ========================================
# ==================================================================================================

print("=================================================")
print("=============== train info prints ===============")
print("=================================================")

print("Top frequency words range :", topN)
print("Embedding Vectors dimension :", embedding_dim)

print("Epoch Size :", epoch_size)
print("Batch Size :", batch_size)

print("Learning Rate :", lr)

print("train x shape :", train_x.shape)
print("train y shape :", train_y.shape)

print("test x shape :", test_x.shape)
print("test y shape :", test_y.shape)

print("model info :", model)

# ==================================================================================================
# ===================================== make batch data loader =====================================
# ==================================================================================================

print("=================================================")
print("================== train start ==================")
print("=================================================")

train_x = torch.LongTensor(train_x)
train_y = torch.FloatTensor(train_y)

test_x = torch.LongTensor(test_x)
test_y = torch.FloatTensor(test_y)

train_set = TensorDataset(train_x, train_y)
test_set = TensorDataset(test_x, test_y)

train_loader = DataLoader(train_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size, shuffle=True)


for epoch_i in range(epoch_size):
    epoch_start_time = time.time()

    losses_per_iter = []
    acc_per_iter = []
    for (batch_x, batch_y) in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred_y = model(batch_x)

        # initialize optimizer to make gradient be zero.
        optimizer.zero_grad()

        iter_loss = loss(pred_y, batch_y.view(-1, 1))
        losses_per_iter.append(iter_loss.item())

        pred_count = pred_y.detach().cpu().numpy() >= 0.5
        ans_count = batch_y.view(-1, 1).cpu().numpy() >= 0.5
        acc = np.mean(pred_count == ans_count)
        acc_per_iter.append(acc)

        iter_loss.backward()
        optimizer.step()
    per_epoch_time = time.time() - epoch_start_time
    print(
        "[Epoch %03d/%03d] - time taken: %.3f | Loss: %.3f | Acc: %.3f"
        % (epoch_i + 1, epoch_size, per_epoch_time, torch.mean(torch.FloatTensor(losses_per_iter)), np.mean(acc_per_iter))
    )

