import torch

class Inference:
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config

    def run(self):
        classes = [
            'T-shirt/top',
            'Trouser',
            'Pullover',
            'Dress',
            'Coat',
            'Sandal',
            'Shirt',
            'Sneaker',
            'Bag',
            'Ankle boot',
        ]

        self.model.eval()
        x, y = self.data['test'][0][0], self.data['test'][0][1]
        with torch.no_grad():
            x = x.to(self.config.device)
            pred = self.model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f"Predicted: '{predicted}', Actual: '{actual}'")
