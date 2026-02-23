from torch.utils.data import DataLoader
from torchvision import datasets

class DataLoaderAdv(object):
    def __init__(self,
                 dataset: datasets,
                 batch_size: int,
                 shuffle: bool,
                 num_workers: int = None):
        self.dataloader_adv = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )