import scipy
import torch
import torch.optim as optim

from Models import SUFCN, SUCNN, SUFormer
from Utils.Losses import SADLoss, SparseLoss
from Utils.Seed_Everything import seed_everything
from Utils.Dataset import Dataset

import numpy as np


def stable(dataloader, seed):
    seed_everything(seed)
    return dataloader


def main(training_data, end_m, lr=0.001, wd=5e-3, lamda=0.001, epochs=100, hidden_dim=128, num_head=8,
         pt_model=None, seed=42, device='cuda', use_nonlinear=False):
    n, num_bands = training_data.shape
    num_end, num_bands = end_m.shape

    dataset = Dataset(training_data)
    train_dataloader = torch.utils.data.DataLoader(dataset, int(n / 10), drop_last=True, shuffle=True)

    # Define Model
    model = SUFormer(num_end=num_end,
                     num_bands=num_bands,
                     hidden_dim=hidden_dim,
                     num_head=num_head).to(device)

    pre_keys = []
    if pt_model is not None:
        state_dict = {}
        model_dict = model.state_dict()
        pretrain_model_para = torch.load(pt_model, map_location=device)
        for k, v in pretrain_model_para.items():
            if k in model_dict.keys():
                state_dict[k] = v
                pre_keys.append(k)
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        print('pretraining model loaded')

    opt_parameters = []
    for name, param in model.named_parameters():
        if 'nonlinear' in name:
            param.requires_grad = True
            params = {"params": param, 'lr': lr * 0.01}
        elif name in pre_keys:
            param.requires_grad = True
            params = {"params": param, 'lr': lr}
        else:
            params = {"params": param, 'lr': lr}
        opt_parameters.append(params)

    optimizer = optim.AdamW(opt_parameters, lr, weight_decay=wd)

    sparsity_loss = SparseLoss(reduction='mean')
    reconstr_loss = SADLoss()
    reconstr_loss2 = torch.nn.MSELoss()

    end_m = torch.tensor(end_m).float().to(device)

    for epoch in range(epochs):
        model.train()
        for x in stable(train_dataloader, seed * epochs + epoch):
            x = x.float().to(device)
            pred, pred_img = model(x, end_m)

            if use_nonlinear:
                loss_recon = reconstr_loss(pred_img, x) + reconstr_loss2(pred_img, x)
            else:
                loss_recon = reconstr_loss(pred @ end_m, x) + reconstr_loss2(pred @ end_m, x)
            loss_sparse = sparsity_loss(pred)
            loss = loss_recon + lamda * loss_sparse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_data = torch.tensor(training_data).float().to(device)
        pred, pred_img = model(test_data, end_m)
        a = pred.detach().cpu().numpy()
        a[a < 0.01] = 0
        a = a / (np.sum(a, axis=1, keepdims=True) + 1e-6)
    return a


if __name__ == "__main__":
    src_dir = r"Data\jasperRidge2_R198.mat"
    label_path = r"Data\end4.mat"
    lib_path = r"Data\lib_Jasper_cut.npy"

    training_data = scipy.io.loadmat(src_dir)
    h = training_data['nRow'][0][0]
    w = training_data['nCol'][0][0]

    training_data = training_data['Y'].T
    training_data = (training_data - training_data.min()) / (training_data.max() - training_data.min())

    lib_data = np.load(lib_path)
    lib_data = (lib_data - lib_data.min()) / (lib_data.max() - lib_data.min())

    label_data = scipy.io.loadmat(label_path)
    label = label_data['M'].T

    end_m = np.concatenate([label, lib_data], axis=0)

    pt_model_name = r'SUFormer_PT.pkl'

    learning_rate = 5e-5
    weight_decay = 5e-3
    lamda = 1e-1
    epochs = 100
    hidden_dim = 128
    num_head = 8  # For SUFormer
    seed = 42
    device = torch.device('cuda')
    use_nonlinear_decoder = True

    seed_everything(seed)
    pred = main(training_data,
                end_m,
                lr=learning_rate,
                wd=weight_decay,
                lamda=lamda,
                epochs=epochs,
                hidden_dim=hidden_dim,
                num_head=num_head,
                pt_model=pt_model_name,
                seed=seed,
                device=device,
                use_nonlinear=use_nonlinear_decoder,)

    a_t = label_data['A']

    pred = pred.T

    rmse_a = np.sqrt(np.mean((pred[:label.shape[0]] - a_t) ** 2))
    rmse = np.sqrt(np.mean((pred[:label.shape[0]] - a_t) ** 2, axis=1))
    print('Overall RMSE: ', rmse_a)
    print('RMSE: ', rmse)