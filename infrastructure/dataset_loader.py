from torchvision import datasets
from torchvision.transforms import ToTensor

class DatasetLoader:
    def __init__(self, config):
        self.config = config
        self.data = {}

    def load_training_data(self):
        training_data = datasets.FashionMNIST(
            root='data',
            train=True,
            download=True,
            transform=ToTensor(),
        )

        self.data['train'] = training_data

    def load_test_data(self):
        test_data = datasets.FashionMNIST(
            root='data',
            train=False,
            download=True,
            transform=ToTensor(),
        )

        self.data['test'] = test_data

    def get_data(self):
        return self.data

    def get_data_by_name(self, name):
        try:
            return self.data[name]
        except KeyError:
            print("There is no data named: {name}")

