import json
import torch

class Configuration:
    def __init__(self, config_file=None):
        if config_file is None:
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        f = open(config_file)
        config = json.load(f)

        self.batch_size = config['batch_size']
        self.device = self.get_device(config['device'])
        self.epochs = config['epochs']

    @staticmethod
    def get_device(device):
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'mps'
        if device == 'mps' and not torch.backends.mps.is_available():
            device = 'cpu'

        return device
