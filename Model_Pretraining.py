import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from Models import SUFCN, SUCNN, SUFormer
from Utils.Seed_Everything import seed_everything

import numpy as np


def main(train_data, lr=0.001, wd=0.005, epochs=500, hidden_dim=128, num_head=8, model_name=None, seed=42, device='cuda'):
    num_end, num_bands = train_data.shape

    # Define Model
    model = SUFormer(num_end=num_end,
                     num_bands=num_bands,
                     hidden_dim=hidden_dim,
                     num_head=num_head).to(device)

    optimizer = optim.AdamW(model.parameters(), lr, weight_decay=wd)
    CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.01 * lr)

    cls_loss = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        seed_everything(seed + epoch)

        abs = np.zeros((num_end * 8, num_end - 1))

        rand = np.random.rand(num_end * 8, 3) + 0.1
        rand_sample = np.random.rand(num_end * 8, 1) + 0.1

        abs[1::8, :1] = rand[1::8, :1]
        abs[2::8, :1] = rand[2::8, :1]
        abs[3::8, :2] = rand[3::8, :2]
        abs[4::8, :2] = rand[4::8, :2]
        abs[5::8, :2] = rand[5::8, :2]
        abs[6::8, :2] = rand[6::8, :2]
        abs[7::8, :2] = rand[7::8, :2]

        for i in range(num_end * 8):
            abs[i] = np.random.permutation(abs[i])

        abs = np.insert(abs, [0], rand_sample, axis=1)
        abs = abs / (np.sum(abs, axis=1, keepdims=True) + 1e-6)

        for i in range(num_end):
            abs_cut = abs[i * 8: (i + 1) * 8]
            abs_cut = np.roll(abs_cut, i, axis=1)
            abs[i * 8: (i + 1) * 8] = abs_cut

        scale = np.random.uniform(0.5, 1., (train_data.shape[0], 1))
        train_data_scaled = train_data * scale

        x = abs @ train_data_scaled
        x = x + np.random.normal(0, 0.005, x.shape)
        x = (x - x.min()) / (x.max() - x.min())
        x = torch.tensor(x, device=device).float()

        pred = model.Encoder(x)

        label_t = torch.tensor(abs, device=device).float()
        loss = cls_loss(pred, label_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = pred.softmax(-1).detach().cpu().numpy()
        acc = np.mean(np.sqrt(np.sum((pred - abs) ** 2, axis=1)))
        CosineLR.step(epoch)

    torch.save(model.state_dict(), model_name)
    print('Training Finished!')


if __name__ == "__main__":
    label_path = r"Data\end4.mat"
    lib_path = r"Data\lib_Jasper_cut.npy"

    lib_data = np.load(lib_path)
    lib_data = (lib_data - lib_data.min()) / (lib_data.max() - lib_data.min())

    label = scipy.io.loadmat(label_path)
    label = label['M'].T

    train_data = np.concatenate([label, lib_data], axis=0)

    model_name = r'SUFormer_PT.pkl'

    learning_rate = 5e-4
    weight_decay = 0.005
    epochs = 500
    hidden_dim = 128
    num_head = 8  # For SpecFormer
    seed = 42
    device = torch.device('cuda:0')

    seed_everything(seed)
    main(train_data,
         lr=learning_rate,
         wd=weight_decay,
         epochs=epochs,
         hidden_dim=hidden_dim,
         num_head=num_head,
         model_name=model_name,
         seed=seed,
         device=device,
         )