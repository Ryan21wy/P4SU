import scipy
import torch
import torch.optim as optim

from Models import SpecFCN, SpecCNN, SpecFormer
from Utils.Losses import SADLoss, SparseLoss
from Utils.Seed_Everything import seed_everything
from Utils.Dataset import Dataset

import numpy as np


def stable(dataloader, seed):
    seed_everything(seed)
    return dataloader


def main(lr=0.001, wd=5e-3, lamda=0.001, epochs=100, hidden_dim=128, num_head=8, pt_model=None, seed=42):
    src_dir = r"Data\jasperRidge2_R198.mat"
    label_path = r"Data\end4.mat"
    lib_path = r"Data\lib_Jasper_cut.npy"

    device = torch.device('cuda')

    training_data = scipy.io.loadmat(src_dir)
    h = training_data['nRow'][0][0]
    w = training_data['nCol'][0][0]

    training_data = training_data['Y'].T
    training_data = (training_data - training_data.min()) / (training_data.max() - training_data.min())
    n, num_bands = training_data.shape

    lib_data = np.load(lib_path)
    lib_data = (lib_data - lib_data.min()) / (lib_data.max() - lib_data.min())

    label_data = scipy.io.loadmat(label_path)
    label = label_data['M'].T

    end_m = np.concatenate([label, lib_data], axis=0)
    end_m = torch.tensor(end_m).float().to(device)
    num_end, num_bands = end_m.shape

    a_t = label_data['A']

    dataset = Dataset(training_data)
    train_dataloader = torch.utils.data.DataLoader(dataset, int(n / 10), drop_last=True, shuffle=True)

    # Define Model
    model = SpecFormer(num_end=num_end,
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

    optimizer = optim.AdamW(model.parameters(), lr, weight_decay=wd)

    sparsity_loss = SparseLoss(reduction='mean')
    reconstr_loss = SADLoss()

    for epoch in range(epochs):
        model.train()
        for x in stable(train_dataloader, seed * epochs + epoch):
            x = x.float().to(device)
            pred = model(x)

            loss_recon = reconstr_loss(pred @ end_m, x)
            loss_sparse = sparsity_loss(pred)
            loss = loss_recon + lamda * loss_sparse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_data = torch.tensor(training_data).float().to(device)
        pred = model(test_data)
        a = pred.detach().cpu().numpy()
        a[a < 0.01] = 0
        a = a / (np.sum(a, axis=1, keepdims=True) + 1e-6)
        a = a.T

        rmse_a = np.sqrt(np.mean((a[:label.shape[0]] - a_t) ** 2))
        rmse = np.sqrt(np.mean((a[:label.shape[0]] - a_t) ** 2, axis=(1)))
        print('Overall RMSE: ', rmse_a)
        print('RMSE: ', rmse)


if __name__ == "__main__":
    pt_model_name = r'SpecFormer_PT.pkl'

    learning_rate = 1e-4
    weight_decay = 5e-3
    lamda = 1e-2
    epochs = 100
    hidden_dim = 128
    num_head = 8  # For SpecFormer
    seed = 42

    seed_everything(seed)
    main(lr=learning_rate,
         wd=weight_decay,
         lamda=lamda,
         epochs=epochs,
         hidden_dim=hidden_dim,
         num_head=num_head,
         pt_model=pt_model_name,
         seed=seed)