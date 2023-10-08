# from model import *
# from utils import *
from torch.utils.data import Dataset, DataLoader

class ProDataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self,dataSet):
        self.dataSet = dataSet#list:[[smile,seq,label],....]
        self.len = len(dataSet)
        self.properties = [int(x[2]) for x in dataSet]# labels
        self.property_list = list(sorted(set(self.properties)))

    def __getitem__(self, index):
        smiles,seq,label = self.dataSet[index]
        return smiles,seq, int(label)

    def __len__(self):
        return self.len

    def get_properties(self):
        return self.property_list

    def get_property(self, id):
        return self.property_list[id]

    def get_property_id(self, property):
        return self.property_list.index(property)
    

