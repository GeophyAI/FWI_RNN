import torch

class CNN(nn.Module):
    def __init__(self,):
        super(cnn, self).__init__()
        pass

        self.conv2d = torch.nn.Conv2d(1, 1, 3, padding=1)
        self._init_params()

    def forward(self, x):
        return self.conv2d(x)
