import os
import numpy as np
import time
import sys
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import DNN, Res_CNN, Bi_GRU_layer_2, BILSTM_withAttention2layer, Multi_Channel_CNN

from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = True

"""
학습을 진행하는 파이썬 파일입니다.
이전에 파이토치 프레임워크를 사용해서 딥러닝을 학습해본 경험이 있어서 참고 없이 스스로 구현했습니다.
학습을 진행하기 위해,
train_x_for_top_10000_maxLen_80.npy
train_y_for_top_10000_maxLen_80.npy
test_x_for_top_10000_maxLen_80.npy
test_y_for_top_10000_maxLen_80.npy
이 필요합니다. utils.py를 한번 실행시켜서 해당 데이터셋을 생성하고 train.py를 실행해야 합니다.
"""

# ==================================================================================================
# ======================================== Hyper Parameters ========================================
# ==================================================================================================

topN = 10000
embedding_dim = 100
vector_len = 80

epoch_size = 20
batch_size = 500

# model = DNN(vocab_size=topN + 2, embedding_dim=embedding_dim, vector_len=vector_len).to(device)
# model = Res_CNN(vocab_size=topN + 2, embedding_dim=embedding_dim, vector_len=vector_len, filter_size=32).to(device)
# model = Bi_GRU_layer_2(vocab_size=topN + 2)
# model = BILSTM_withAttention2layer(vocab_size=topN + 2, embedding_dim=embedding_dim, vector_len=vector_len, unit_num=128).to(device)
# model = Multi_Channel_CNN(
#     vocab_size=topN + 2,
#     embedding_dim=embedding_dim,
#     vector_len=vector_len,
#     n_filters_a=n_filters_a,
#     n_filters_b=n_filters_b,
#     filter_sizes_a=filter_sizes_a,
#     filter_sizes_b=filter_sizes_b,
#     dropout=dropout,
# ).to(device)

lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)
loss = torch.nn.BCELoss().to(device)

dataPath = "./dataSet"
train_x = np.load(os.path.join(dataPath, "train_x_for_top_%d_maxLen_%d.npy" % (topN, vector_len)))
train_y = np.load(os.path.join(dataPath, "train_y_for_top_%d_maxLen_%d.npy" % (topN, vector_len)))
test_x = np.load(os.path.join(dataPath, "test_x_for_top_%d_maxLen_%d.npy" % (topN, vector_len)))
test_y = np.load(os.path.join(dataPath, "test_y_for_top_%d_maxLen_%d.npy" % (topN, vector_len)))

# ==================================================================================================
# ======================================== print train info ========================================
# ==================================================================================================

print("================================================")
print("================== train info ==================")
print("================================================")

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
# ========================================== model train  ==========================================
# ==================================================================================================

print("================================================")
print("================= train start ==================")
print("================================================")

acc_per_epoch = []
loss_per_epoch = []
val_acc_per_epoch = []
val_loss_per_epoch = []

# divide train set, validation set to 8:2 ratio.
divide_idx = int(train_x.shape[0] * 0.8)

validation_x = torch.LongTensor(train_x[divide_idx:,])
validation_y = torch.FloatTensor(train_y[divide_idx:,])

train_x = torch.LongTensor(train_x[:divide_idx,])
train_y = torch.FloatTensor(train_y[:divide_idx,])

test_x = torch.LongTensor(test_x)
test_y = torch.FloatTensor(test_y)

train_set = TensorDataset(train_x, train_y)
validation_set = TensorDataset(validation_x, validation_y)
test_set = TensorDataset(test_x, test_y)

train_loader = DataLoader(train_set, batch_size, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size, shuffle=True)

loss_hist = []
acc_hist = []
val_loss_hist = []
val_acc_hist = []


for epoch_i in range(epoch_size):
    model.train()
    epoch_start_time = time.time()

    losses_per_epoch = []
    acc_per_epoch = []
    for (batch_x, batch_y) in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred_y = model(batch_x)

        # initialize optimizer to make gradient be zero.
        optimizer.zero_grad()

        # calculate loss value per iter.
        iter_loss = loss(pred_y, batch_y.view(-1, 1))
        losses_per_epoch.append(iter_loss.item())

        # calculate gradient and
        iter_loss.backward()
        optimizer.step()

        # calculate accuary per iter.
        pred_count = pred_y.detach().cpu().numpy() >= 0.5
        ans_count = batch_y.view(-1, 1).cpu().numpy() == 1
        train_acc = np.mean(pred_count == ans_count)
        acc_per_epoch.append(train_acc)

    per_epoch_time = time.time() - epoch_start_time

    with torch.no_grad():
        model.eval()
        losses_per_validation = []
        acc_per_validation = []
        for (batch_x, batch_y) in validation_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred_y = model(batch_x)

            # calculate loss value per iter.
            iter_loss = loss(pred_y, batch_y.view(-1, 1))
            losses_per_validation.append(iter_loss.item())

            # calculate accuary per iter.
            pred_count = pred_y.detach().cpu().numpy() >= 0.5
            ans_count = batch_y.view(-1, 1).cpu().numpy() == 1
            validation_acc = np.mean(pred_count == ans_count)
            acc_per_validation.append(validation_acc)
    print(
        f"[Epoch {epoch_i + 1:03}/{epoch_size:03}] - time taken: {per_epoch_time:.3f} | Loss: {np.mean(losses_per_epoch):.3f} | Acc: {np.mean(acc_per_epoch) * 100:.3f} | validation Loss: {np.mean(losses_per_validation):.3f} | validation Acc : {np.mean(acc_per_validation) * 100:.3f}"
    )

    loss_hist.append(np.mean(losses_per_epoch))
    acc_hist.append(np.mean(acc_per_epoch) * 100)
    val_loss_hist.append(np.mean(losses_per_validation))
    val_acc_hist.append(np.mean(acc_per_validation) * 100)


model.eval()
test_start_time = time.time()

losses_per_iter = []
acc_per_iter = []
for (batch_x, batch_y) in test_loader:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    pred_y = model(batch_x)

    # calculate loss value per iter.
    iter_loss = loss(pred_y, batch_y.view(-1, 1))
    losses_per_iter.append(iter_loss.item())

    # calculate accuary per iter.
    pred_count = pred_y.detach().cpu().numpy() >= 0.5
    ans_count = batch_y.view(-1, 1).cpu().numpy() == 1
    acc = np.mean(pred_count == ans_count)
    acc_per_iter.append(acc)

total_test_time = time.time() - test_start_time
print(f"[{'model test':^13s}] - time taken: {total_test_time:.3f} | Loss: {np.mean(losses_per_iter):.3f} | Acc: {np.mean(acc_per_iter) * 100:.3f}")

print("================================================")
print("========= model training is all done ===========")
print("================================================")

# ==================================================================================================
# =================================== plot train result figures +===================================
# ==================================================================================================

plt.figure(figsize=(16, 8))

x = np.arange(1, epoch_size + 1, 1)

plt.subplot(121)
plt.plot(x, loss_hist)
plt.plot(x, val_loss_hist)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid()
plt.legend(["loss", "validation loss"])

plt.subplot(122)
plt.plot(x, acc_hist)
plt.plot(x, val_acc_hist)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.grid()
plt.legend(["accuracy", "validation accuracy"])

plt.show()
