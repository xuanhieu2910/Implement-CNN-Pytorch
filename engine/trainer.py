import torch
import time
from torch import nn, optim
from torch.utils.data import DataLoader
import utils.metrics as metrics
import utils.utils as utils
import tqdm as tqdm

class Trainer():
    def __init__(self,
                 epochs: int,
                 data_train_loader: DataLoader ,
                 data_test_loader: DataLoader,
                 model: nn.Module,
                 loss: nn.Module,
                 optimizer: optim.Optimizer,
                 device: torch.device
                 ):
        self.epochs = epochs
        self.data_train_loader = data_train_loader
        self.data_test_loader = data_test_loader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device


    def training_step(self):
        train_time_start_on_cpu = time.time()
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch} ------------------------------")

            training_loss = 0.0
            for batch, (X, y) in enumerate(self.data_train_loader):
                self.model.train()
                #Forward pass
                y_pred = self.model(X)
                #Loss
                loss = self.loss(y_pred, y)
                training_loss += loss
                self.optimizer.zero_grad()
                #Backward
                loss.backward()
                #Optimizer
                self.optimizer.step()
                # Print out how many samples have been seen
                if batch % 400 == 0:
                    print(f"Looked at {batch * len(X)}/{len(self.data_train_loader.dataset)} samples")

            training_loss /= len(self.data_train_loader)
            print(f"Training Loss: {training_loss}")
        end_time_train_on_cpu = time.time()
        utils.print_train_time(train_time_start_on_cpu, end_time_train_on_cpu, self.device)

    def test_step(self):
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch} ------------------------------")
            test_loss = 0.0
            acc_test = 0.0
            self.model.eval()

            with torch.inference_mode():
                for X, y in self.data_test_loader:
                    test_pred = self.model(X)
                    test_loss += self.loss(test_pred, y)
                    acc_test += metrics.accuracy_fn(test_pred.argmax(dim=1), y)

                test_loss /= len(self.data_test_loader)
                acc_test /= len(self.data_test_loader)
            print(f"Test loss: {test_loss:.5f}, Test acc: {acc_test:.2f}%\n")




