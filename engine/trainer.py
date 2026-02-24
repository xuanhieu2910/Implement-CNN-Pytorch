from timer import timer
from torch import nn, optim
from torch.utils.data import DataLoader
import tqdm as tqdm

class Trainer():
    def __init__(self,
                 epochs: int,
                 data_train_loader: DataLoader ,
                 data_test_loader: DataLoader,
                 model: nn.Module,
                 loss: nn.Module,
                 optimizer: optim.Optimizer,
                 ):
        self.epochs = epochs
        self.data_train_loader = data_train_loader
        self.data_test_loader = data_test_loader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer


    def training_step(self):
        train_time_start_on_cpu = timer()

        for epoch in range(self.epochs):
            print(f"Epoch: {epoch} ------------------------------")

            training_loss = 0.0
            for batch, (X, y) in enumerate(self.data_train_loader):
                print("Batch: ", batch)
                print("X: ", X.size())
                print("y: ", y.size())
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
            training_loss /= len(self.data_train_loader)
            print(f"Training Loss: {training_loss}")
        print("Shut down")

