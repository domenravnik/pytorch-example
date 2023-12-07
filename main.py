import torch

from infrastructure.dataset_loader import DatasetLoader
from networks.neural_network import NeuralNetwork
from infrastructure.configuration import Configuration
from training import Training
from inference import Inference

mode = 'training'  # ['training', 'inference']

config = Configuration('config.json')

dataset_loader = DatasetLoader(config)
dataset_loader.load_training_data()
dataset_loader.load_test_data()
data = dataset_loader.get_data()

if mode == 'training':
    training = Training(data, config)
    model = training.run()

    torch.save(model.state_dict(), 'models/model.pth')
    print("Saved PyTorch model state to model.pth")

elif mode == 'inference':
    model = NeuralNetwork().to(config.device)
    model.load_state_dict(torch.load('models/model.pth'))

    inference = Inference(model, data, config)
    inference.run()
