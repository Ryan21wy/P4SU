import numpy as np
import scipy

import os
import random
import torch
import torch.optim as optim

from Compared_Methods.SUnSAL import SUnSAL
from Compared_Methods.CLSUnSAL import CLSUnSAL
from Compared_Methods.S2WSU import S2WSU
from Compared_Methods.SUnCNN import SUnCNN


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module


def SAM(x, y):
    x_norm = x / np.sqrt(np.sum(x ** 2, axis=1, keepdims=True))
    y_norm = y / np.sqrt(np.sum(y ** 2, axis=1, keepdims=True))
    s = np.dot(x_norm, y_norm.T)
    th = np.arccos(s)
    return th


def main(lamda=1., iter=1000):
    src_dir = r"Data\jasperRidge2_R198.mat"
    label_path = r"Data\end4.mat"
    lib_path = r"Data\lib_Jasper_cut.npy"

    mat_data = scipy.io.loadmat(src_dir)
    print(mat_data.keys())
    test_data = mat_data['Y'].T

    h = mat_data['nRow'][0][0]
    w = mat_data['nCol'][0][0]

    test_data = (test_data - test_data.min()) / (test_data.max() - test_data.min())
    n, num_bands = test_data.shape

    lib_data = np.load(lib_path)
    lib_data = (lib_data - lib_data.min()) / (lib_data.max() - lib_data.min())

    label_data = scipy.io.loadmat(label_path)
    label = label_data['M'].T
    a_t = label_data['A']

    end_m = np.concatenate([label, lib_data], axis=0)
    print(end_m.shape)

    # model = CLSUnSAL(AL_iters=iter, lambd=lamda)
    # model = S2WSU(AL_iters=iter, lambd=lamda)
    model = SUnSAL(AL_iters=iter, lamda=lamda)
    a = model.compute_abundances(test_data.T, end_m.T, h, w)

    a = a / (np.sum(a, axis=0, keepdims=True) + 1e-6)
    a[a < 0.01] = 0
    a = a / (np.sum(a, axis=0, keepdims=True) + 1e-6)
    a = a.reshape(-1, h, w).transpose(0, 2, 1)

    a_t = a_t.reshape(-1, h, w).transpose(0, 2, 1)

    rmse_a = np.sqrt(np.mean((a[:label.shape[0]] - a_t) ** 2))
    rmse = np.sqrt(np.mean((a[:label.shape[0]] - a_t) ** 2, axis=(1, 2)))
    print('Overall RMSE: ', rmse_a)
    print('RMSE: ', rmse)


def main2(lr=1e-3, iter=10000, lamda=1e-3, exp_weight=0.99, seed=42):
    src_dir = r"Data\jasperRidge2_R198.mat"
    label_path = r"Data\end4.mat"
    lib_path = r"Data\lib_Jasper_cut.npy"

    mat_data = scipy.io.loadmat(src_dir)
    print(mat_data.keys())
    test_data = mat_data['Y']

    h = mat_data['nRow'][0][0]
    w = mat_data['nCol'][0][0]

    test_data = (test_data - test_data.min()) / (test_data.max() - test_data.min())
    num_bands, n = test_data.shape

    lib_data = np.load(lib_path)
    lib_data = (lib_data - lib_data.min()) / (lib_data.max() - lib_data.min())

    label_data = scipy.io.loadmat(label_path)
    label = label_data['M'].T
    a_t = label_data['A']

    end_m = np.concatenate([label, lib_data], axis=0)
    num_ends = end_m.shape[0]

    device = torch.device('cuda')

    model = SUnCNN(niters=iter, lr=lr)
    model.L = num_bands
    model.M = num_ends
    model.init_architecture(seed=seed)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr)
    reconstr_loss = torch.nn.MSELoss()

    img_noisy_np = torch.rand((num_bands, h, w))

    end_mt = torch.tensor(end_m).float().to(device)
    target = torch.tensor(test_data).float().to(device)

    model.train()
    for i in range(iter):
        x = img_noisy_np.float().to(device)
        pred = model(x.unsqueeze(0))
        pred = pred.squeeze(0).flatten(1)
        loss = reconstr_loss(end_mt.T @ pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i == 0:
            pred_avg = pred.detach().cpu().numpy()
        else:
            pred_avg = pred_avg * exp_weight + pred.detach().cpu().numpy() * (1 - exp_weight)

        a = pred_avg.copy()
        a[a < 0.01] = 0
        a = a / (np.sum(a, axis=0, keepdims=True) + 1e-6)

    rmse_a = np.sqrt(np.mean((a[:label.shape[0]] - a_t) ** 2))
    rmse = np.sqrt(np.mean((a[:label.shape[0]] - a_t) ** 2, axis=1))
    print('Overall RMSE: ', rmse_a)
    print('RMSE: ', rmse)


if __name__ == "__main__":
    # For SUnSAL, CLSUnSAL, S2WSU
    seed = 42
    lamda = 1e-3
    iter = 1000
    seed_everything(seed)
    main(lamda=lamda, iter=iter)

    # For SUnCNN
    seed = 42
    lr = 1e-3
    iter = 20000
    exp_weight = 0.99
    lamda = 1e-3
    seed_everything(seed)
    main2(lr=lr, iter=iter, lamda=lamda, exp_weight=exp_weight, seed=seed)
