import PIL.Image as Image
import torch
import torch.nn as nn
from torchvision.models import AlexNet_Weights
from torch.utils.data import Dataset
import pandas as pd
import glob


def get_df(path, classes=['dogs', 'cats']):
    paths = pd.DataFrame({'class': [], 'path': []})
    for c in classes:
        df = pd.DataFrame({
            'class': c,
            'path': glob.glob(path + c + '/*')
        })

        paths = pd.concat([paths, df])

    paths.reset_index(inplace=False)

    return paths


class CatDogDataset(Dataset):
    def __init__(self, df, classes, transform=None):
        self.paths = df
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        row = self.paths.iloc[idx]
        img = Image.open(row['path'])
        if self.transform is not None:
            return self.transform(img), self.classes[row['class']]
        else:
            return img, self.classes[row['class']]


def get_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=AlexNet_Weights.DEFAULT)
    model.features.requires_grad = False
    model.classifier = nn.Sequential(
        *model.classifier[:-1],
        nn.Linear(4096, 2, bias=True)
    )
    return model