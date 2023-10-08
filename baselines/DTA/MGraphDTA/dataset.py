import torch
from torch_geometric.data import InMemoryDataset

class GNNDataset(InMemoryDataset):

    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        if train:
            self.data, self.slices = torch.load(self.processed_paths[0]) #self.processed_paths[0]  ===> train.pt
        else:
            self.data, self.slices = torch.load(self.processed_paths[1]) #self.processed_paths[1]  ===> test.pt

    @property
    def raw_file_names(self):
        return ['train.csv', 'test.csv']

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def process(self):
        pass

if __name__ == "__main__":
    dataset = GNNDataset('data/davis')


