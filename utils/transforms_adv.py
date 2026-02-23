from torchvision import transforms

def transform_2tensor():
    return transforms.Compose(
        [transforms.ToTensor()]
        )