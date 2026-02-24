from typing import Optional

from torch.utils.data import DataLoader
from torchvision import datasets

class DataLoaderAdv(object):
    def __init__(self,
                 dataset_train: datasets,
                 dataset_test: datasets,
                 batch_size: Optional[int] = 32,
                 shuffle: Optional[bool] = True):
        self.train_dataloader = DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            shuffle=shuffle
        )
        self.test_dataloader = DataLoader(
            dataset=dataset_test,
            batch_size=batch_size,
            shuffle=shuffle
        )


    def __str__(self):
        return (
                f"DataLoader {self.train_dataloader.__str__()} " +
                f"{self.test_dataloader.__str__()}" +
                f"Length of train_dataloader: {len(self.train_dataloader)} - Batch size: {self.train_dataloader.batch_size} " +
                f"Length of test_dataloader: {len(self.test_dataloader)} - Batch size: {self.test_dataloader.batch_size}"
                )
