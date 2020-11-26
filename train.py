import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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

lr = 0.001

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

# ==================================================================================================
# ===================================== make batch data loader =====================================
# ==================================================================================================

train_x = torch.LongTensor(train_x)
train_y = torch.FloatTensor(train_y)

test_x = torch.LongTensor(test_x)
test_y = torch.FloatTensor(test_y)

train_set = TensorDataset(train_x, train_y)
test_set = TensorDataset(test_x, test_y)

train_loader = DataLoader(train_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size, shuffle=True)

for (batch_x, batch_y) in train_loader:
    print("batch_x.shape :", batch_x.shape)
    print("batch_y.shape :", batch_y.shape)
    break
