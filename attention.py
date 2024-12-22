import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

def spatial_attention_map (feature, label, autoencoder, models, criterion):
    feature.requires_grad_()
    loss = 0
    decode = autoencoder.module.decode(feature)
    decode = normalize(decode)

    for model in models:
         output = model(decode)
         loss += criterion(output, label)

    loss /= len(models)

    grad = torch.autograd.grad(loss, feature)[0]
    grad = torch.abs(grad)

    sam = torch.sigmoid(grad)
    feature = feature.detach()

    return feature, sam


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and labels.csv file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.labels_path = os.path.join(root_dir, 'labels.csv')
        self.labels_df = pd.read_csv(self.labels_path, header=None, names=['filename', 'label', 'targeted_label'])
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, self.labels_df.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(int(self.labels_df.iloc[idx, 1]))
        label2 = torch.tensor(int(self.labels_df.iloc[idx, 2]))

        if self.transform:
            image = self.transform(image)


        return image, label

