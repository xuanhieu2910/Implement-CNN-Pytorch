import torchvision.datasets as datasets


class FashionMNIST():

    def __init__(self, root,
                 download = True,
                 transforms = None):
        self.root = root
        self.download = download
        self.transforms = transforms
        self.training_data = None
        self.validation_data = None

    def download_dataset(self):
        if self.download:
            self.training_data = datasets.FashionMNIST(
                root=self.root,
                train=True,
                download=True,
                transform=self.transforms
            )

            self.validation_data = datasets.FashionMNIST(
                root=self.root,
                train=False,
                download=True,
                transform=self.transforms
            )
        return self.training_data, self.validation_data