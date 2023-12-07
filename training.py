import torch
from torch import nn
from torch.utils.data import DataLoader

from networks.neural_network import NeuralNetwork
from train import train
from test import test

class Training:
    def __init__(self, data, config):
        self.data = data
        self.config = config

        self.train_dataloader = DataLoader(data['train'], batch_size=self.config.batch_size)
        self.test_dataloader = DataLoader(data['test'], batch_size=self.config.batch_size)

    def run(self):
        print(f"Using {self.config.device} device")

        model = NeuralNetwork().to(self.config.device)
        print(model)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        for t in range(self.config.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(self.train_dataloader, model, loss_fn, optimizer, self.config.device)
            test(self.test_dataloader, model, loss_fn, self.config.device)

        print("Done!")

        return model
