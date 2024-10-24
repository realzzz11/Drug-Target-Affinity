import numpy as np
import torch
from torch.autograd import Variable
from math import sqrt
from scipy import stats
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve


# def batch_pad(arr):
#     N = max([a.shape[0] for a in arr])
#     if arr[0].ndim == 1:
#         new_arr = np.zeros((len(arr), N))
#         new_arr_mask = np.zeros((len(arr), N))
#         for i, a in enumerate(arr):
#             n = a.shape[0]
#             new_arr[i, :n] = a + 1
#             new_arr_mask[i, :n] = 1
#         return new_arr, new_arr_mask

#     elif arr[0].ndim == 2:
#         new_arr = np.zeros((len(arr), N, N))
#         new_arr_mask = np.zeros((len(arr), N, N))
#         for i, a in enumerate(arr):
#             n = a.shape[0]
#             new_arr[i, :n, :n] = a
#             new_arr_mask[i, :n, :n] = 1
#         return new_arr, new_arr_mask
def batch_pad(arr):
    """
    该函数用于对输入数组进行填充操作，以适应不同大小的数据。
    根据输入数据的形状，动态计算填充的维度，确保输入的数据能够被正确填充。

    Args:
        arr (list): 输入列表，每个元素可能是不同维度的数组。

    Returns:
        new_arr: 填充后的数组
        new_arr_mask: 对应的掩码数组
    """
    # 获取最大长度的第一维（时间步/序列长度）
    N = max([a.shape[0] for a in arr])
    # 获取最大长度的第二维（特征维度）
    D = max([a.shape[1] for a in arr]) if arr[0].ndim > 1 else 1

    if arr[0].ndim == 1:  # 如果是 1 维数据
        new_arr = np.zeros((len(arr), N))
        new_arr_mask = np.zeros((len(arr), N))
        for i, a in enumerate(arr):
            n = a.shape[0]
            new_arr[i, :n] = a + 1
            new_arr_mask[i, :n] = 1
        return new_arr, new_arr_mask

    elif arr[0].ndim == 2:  # 如果是 2 维数据，如蛋白质特征
        new_arr = np.zeros((len(arr), N, D))  # 填充到最大长度和最大维度
        new_arr_mask = np.zeros((len(arr), N))  # 只需要记录时间步的掩码
        for i, a in enumerate(arr):
            n = a.shape[0]
            d = a.shape[1]
            new_arr[i, :n, :d] = a  # 填充数据
            new_arr_mask[i, :n] = 1  # 对应位置设为 1，表示有效数据
        return new_arr, new_arr_mask


def fps2number(arr):
    new_arr = np.zeros((arr.shape[0], 1024))
    for i, a in enumerate(arr):
        new_arr[i, :] = np.array(list(a), dtype=int)
    return new_arr


def batch2tensor(batch_data, device):
    atoms_pad, atoms_mask = batch_pad(batch_data[0])
    adjacencies_pad, _ = batch_pad(batch_data[1])
    fps = fps2number(batch_data[2])
    amino_pad, amino_mask = batch_pad(batch_data[3])
    print(atoms_pad.shape, atoms_mask.shape, adjacencies_pad.shape, fps.shape, amino_pad.shape, amino_mask.shape)
    atoms_pad = Variable(torch.LongTensor(atoms_pad)).to(device)
    atoms_mask = Variable(torch.FloatTensor(atoms_mask)).to(device)
    adjacencies_pad = Variable(torch.LongTensor(adjacencies_pad)).to(device)
    fps = Variable(torch.FloatTensor(fps)).to(device)
    amino_pad = Variable(torch.LongTensor(amino_pad)).to(device)
    amino_mask = Variable(torch.FloatTensor(amino_mask)).to(device)

    label = torch.FloatTensor(batch_data[4]).to(device)

    return atoms_pad, atoms_mask, adjacencies_pad, fps, amino_pad, amino_mask, label


def load_data(datadir, target_type):
    if target_type:
        dir_input = datadir + '/' + target_type + '/'
    else:
        dir_input = datadir + '/'
    compounds = np.load(dir_input + 'compounds.npy', allow_pickle=True)
    adjacencies = np.load(dir_input + 'adjacencies.npy', allow_pickle=True)
    fingerprint = np.load(dir_input + 'fingerprint.npy', allow_pickle=True)
    proteins = np.load(dir_input + 'proteins.npy', allow_pickle=True)
    interactions = np.load(dir_input + 'interactions.npy', allow_pickle=True)
    data_pack = [compounds, adjacencies, fingerprint, proteins, interactions]
    return data_pack


def split_data(train_data, ratio=0.1):
    idx = np.arange(len(train_data[0]))
    np.random.shuffle(idx)
    num_dev = int(len(train_data[0]) * ratio)
    idx_dev, idx_train = idx[:num_dev], idx[num_dev:]
    data_train = [train_data[di][idx_train] for di in range(len(train_data))]
    data_dev = [train_data[di][idx_dev] for di in range(len(train_data))]
    return data_train, data_dev


def regression_scores(label, pred):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    rmse = sqrt(((label - pred)**2).mean(axis=0))
    pearson = np.corrcoef(label, pred)[0, 1]
    spearman = stats.spearmanr(label, pred)[0]
    return round(rmse, 6), round(pearson, 6), round(spearman, 6)


def classification_scores(label, pred_score, pred_label):
    label = label.reshape(-1)
    pred_score = pred_score.reshape(-1)
    pred_label = pred_label.reshape(-1)
    auc = roc_auc_score(label, pred_score)
    acc = accuracy_score(label, pred_label)
    precision, recall, _ = precision_recall_curve(label, pred_label)
    aupr = metrics.auc(recall, precision)
    return round(auc, 6), round(acc, 6), round(aupr, 6)
