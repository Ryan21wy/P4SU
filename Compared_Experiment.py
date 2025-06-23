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
from Compared_Methods.KHype import SKHype_L1
from Compared_Methods.FCLS import FCLS
from Compared_Methods.rNMF import robust_nmf


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


def main(test_data, label, lib_data, lamda=1., lamda_l1=0., iter=1000, method='SUnSAL'):
    n, num_bands = test_data.shape
    end_m = np.concatenate([label, lib_data], axis=0)

    if method == 'SUnSAL':
        model = SUnSAL(AL_iters=iter, lamda=lamda)
        a = model.compute_abundances(test_data.T, end_m.T)
    elif method == 'CLSUnSAL':
        model = CLSUnSAL(AL_iters=iter, lamda=lamda)
        a = model.compute_abundances(test_data.T, end_m.T)
    elif method == 'S2WSU':
        model = S2WSU(AL_iters=iter, lamda=lamda)
        a = model.compute_abundances(test_data.T, end_m.T, h, w)
    elif method == 'SK-Hype':
        a = SKHype_L1(test_data.T, end_m.T, lamda, kernel='polynomial', lambda_reg=lamda_l1, max_iter=iter, n_jobs=8)
    elif method == 'rNMF':
        fcls = FCLS()
        a_map = fcls.map(test_data, end_m)
        a_map[a_map < 0] = 0
        a_map = a_map / (np.sum(a_map, axis=1, keepdims=True) + 1e-6)

        user_prov = {}
        user_prov['basis'] = end_m.T
        user_prov['coeff'] = a_map.T
        user_prov['outlier'] = np.random.rand(num_bands, n)

        _, a, _, _ = robust_nmf(test_data.T, end_m.shape[0], 2, lamda, 'user', sum_to_one=False, tol=1e-5, max_iter=iter, user_prov=user_prov, update_end=False)
    else:
        raise ValueError("method should be choose in ['SUnSAL', 'CLSUnSAL', 'S2WSU', 'SK-Hype', 'rNMF']")

    a = a / (np.sum(a, axis=0, keepdims=True) + 1e-6)
    a[a < 0.01] = 0
    a = a / (np.sum(a, axis=0, keepdims=True) + 1e-6)
    a = a.reshape(-1, h, w).transpose(0, 2, 1)
    return a


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
    src_dir = r"Data\jasperRidge2_R198.mat"
    label_path = r"Data\end4.mat"
    lib_path = r"Data\lib_Jasper_cut.npy"

    mat_data = scipy.io.loadmat(src_dir)
    test_data = mat_data['Y'].T

    h = mat_data['nRow'][0][0]
    w = mat_data['nCol'][0][0]

    test_data = (test_data - test_data.min()) / (test_data.max() - test_data.min())

    lib_data = np.load(lib_path)
    lib_data = (lib_data - lib_data.min()) / (lib_data.max() - lib_data.min())

    label_data = scipy.io.loadmat(label_path)
    label = label_data['M'].T

    end_m = np.concatenate([label, lib_data], axis=0)

    # For SUnSAL, CLSUnSAL, S2WSU, SK-Hype, rNMF
    seed = 42
    lamda = 1e-3
    lamda_l1 = 0. # for SK-Hype
    iter = 1000
    seed_everything(seed)
    pred = main(test_data, end_m, lib_data, lamda=lamda, lamda_l1=lamda_l1, iter=iter, method='SUnSAL')

    a_t = label_data['A']

    a_t = a_t.reshape(-1, h, w).transpose(0, 2, 1)

    rmse_a = np.sqrt(np.mean((pred[:label.shape[0]] - a_t) ** 2))
    rmse = np.sqrt(np.mean((pred[:label.shape[0]] - a_t) ** 2, axis=(1, 2)))
    print('Overall RMSE: ', rmse_a)
    print('RMSE: ', rmse)