import torch
import utils.transforms_adv as transform
from utils.utils import load_config
from datasets.fashionmnist import FashionMNIST
from dataloader.dataloader_adv import DataLoaderAdv
from engine.trainer import Trainer
from models.simple_nn import SimpleNN as Model, SimpleNN
from engine.losser import Losser
from engine.optimizer_custom import OptimizerCustom

def main():
    cfg = load_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_training, test_training =  FashionMNIST(
                                    root = ".\datasets\dataset",
                                    transforms=transform.transform_2tensor()).download_dataset()
    data_loader = DataLoaderAdv(data_training, test_training)
    model = SimpleNN(input_dim=784, output_dim=10, hidden_dim=10, device = device)
    loss_fn = Losser()
    optimizer = OptimizerCustom(model.parameters(), 0.001)
    trainer = Trainer(epochs=3,
                      data_train_loader = data_loader.train_dataloader,
                      data_test_loader =  data_loader.test_dataloader,
                      model = model,
                      loss = loss_fn,
                      optimizer = optimizer,
                      device = device)
    trainer.training_step()


if __name__ == "__main__":
    main()