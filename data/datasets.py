import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from PIL import Image
import pandas as pd


class TrainDataset(Dataset):
    def __init__(self, resize, transform):
        metadata_file = './data/train/metadata/metadata.csv'
        self.meta = pd.read_csv(metadata_file, encoding='utf-8')
        self.genre = {key: val for (key, val) in zip(self.meta['label key'], self.meta['genre'])}
        self.genre_count = len(set(self.genre))
        self.resize = resize
        self.transform = transform
        
    def __len__(self):
        # number of samples
        return len(self.meta)
    
    def __getitem__(self, index):
        file = f"./data/train/img/{self.meta.loc[index, 'folder']}/{self.meta.loc[index, 'file']}"
        label = self.meta.loc[index, 'label key']
        img = Image.open(file).convert('RGB')
        if img.size != (720, 1280):
            img = self.resize(img)
        img = self.transform(img)
        return img, label


class TestDataset(Dataset):
    def __init__(self, resize, transform):
        metadata_file = './data/test/metadata/metadata.csv'
        self.meta = pd.read_csv(metadata_file, encoding='utf-8')
        self.genre = {key: val for (key, val) in zip(self.meta['label key'], self.meta['genre'])}
        self.genre_count = len(set(self.genre))
        self.resize = resize
        self.transform = transform
        
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, index):
        file = f"./data/test/img/{self.meta.loc[index, 'folder']}/{self.meta.loc[index, 'file']}"
        label = self.meta.loc[index, 'label key']
        game = self.meta.loc[index, 'game']
        img = Image.open(file).convert('RGB')
        
        # if img.size != (252, 448):
        #     img = self.resize(img)
        
        if img.size != (720, 1280):
            img = self.resize(img)
            
        img = self.transform(img)
        return img, label, game, file


class ValidationDataset(Dataset):
    def __init__(self, resize, transform):
        metadata_file = './data/validation/metadata/metadata.csv'
        self.meta = pd.read_csv(metadata_file, encoding='utf-8')
        self.genre = {key: val for (key, val) in zip(self.meta['label key'], self.meta['genre'])}
        self.genre_count = len(set(self.genre))
        self.resize = resize
        self.transform = transform
        
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, index):
        file = f"./data/validation/img/{self.meta.loc[index, 'folder']}/{self.meta.loc[index, 'file']}"
        label = self.meta.loc[index, 'label key']
        img = Image.open(file).convert('RGB')
        if img.size != (720, 1280):
            img = self.resize(img)
        img = self.transform(img)
        return img, label


class CustomTestDataset(Dataset):
    def __init__(self, resize, transform, file):
        self.file = file
        self.transform = transform
        self.resize = resize

    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        img = Image.open(self.file).convert('RGB')
        if img.size != (720, 1280):
            img = self.resize(img)
        img = self.transform(img)
        return img


# Transformations
data_resize = T.Resize((720, 1280))
data_transform = T.Compose([T.ToImage(), T.ToDtype(torch.float, scale=True)])

# Data
train_data = TrainDataset(data_resize, data_transform)
val_data = ValidationDataset(data_resize, data_transform)
test_data = TestDataset(data_resize, data_transform)

