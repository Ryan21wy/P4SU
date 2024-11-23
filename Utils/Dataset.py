import torch
import torch.utils.data as data
import torchvision.transforms as tvtf

class Dataset(data.Dataset):
    def __init__(self, training_data, transform=tvtf.ToTensor()):
        super(Dataset, self).__init__()
        self.transform = transform
        self.train_data = training_data

    def __getitem__(self, index):
        img = self.train_data[index]
        if self.transform is not None:
            img = torch.tensor(img.copy())
        return img

    def __len__(self):
        return len(self.train_data)