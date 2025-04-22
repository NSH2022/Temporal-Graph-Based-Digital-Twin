from torch_geometric.data import Dataset
import glob2
import torch 
import os



class Graph_Dataset(Dataset):
    def __init__(self, PATH):
        self.PROCESS_FILES = sorted(glob2.glob(PATH + "data__*"))
        super().__init__(PATH)

    def len(self):
        return len(self.PROCESS_FILES)

    def get(self, idx):
        data = torch.load(self.PROCESS_FILES[idx])
        return data
        
