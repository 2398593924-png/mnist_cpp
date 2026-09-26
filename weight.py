import torch
import torch.nn as nn
import numpy as np
 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = x.flatten(start_dim = 0)
        x = self.net(x)
        return x
    
if __name__ == '__main__':
    model = Net()
    model.load_state_dict(torch.load("./weights/mnist_9.pth", map_location="cpu"))

    for name, param in model.named_parameters():
        with open(f"./weights/{name}.txt", "w", encoding="utf-8") as f:
            data = param.detach().cpu().numpy()

            np.savetxt(f, data.reshape(-1, data.shape[-1]) if data.ndim > 1 else data.reshape(1, -1), fmt="%+.8f")
            f.write("\n")
    
    print("Done")
