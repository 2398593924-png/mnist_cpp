import torch
import torch.nn as nn
import torch.nn.functional as F
 
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3801,))
])

train_dataset = MNIST(root='./',
                      train=True,
                      transform=transformer,
                      download=True)

test_dataset = MNIST(root='./',
                      train=False,
                      transform=transformer,
                      download=True)

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
        x = x.flatten(start_dim = 1)
        x = self.net(x)

        return x
    
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TrainLoader = DataLoader(dataset=train_dataset,
                             batch_size=64,
                             shuffle=True,
                             num_workers=2,
                             pin_memory=True)

    TestLoader = DataLoader(dataset=test_dataset,
                             batch_size=64,
                             shuffle=False,
                             num_workers=2,
                             pin_memory=True)
    
    model = Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for _, (img, label) in enumerate(TrainLoader):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()

            pred = model(img)
            loss = criterion(pred, label)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for img, label in TestLoader:
                img, label = img.to(device), label.to(device)
                pred = model(img)
                predicted = pred.argmax(dim=1)
                correct += (predicted == label).sum().item()
                total += label.size(0)

        acc = correct / total
        
        print(f"Epoch {epoch}: Loss = {total_loss / len(TrainLoader)}, Accurate = {acc}")

        torch.save(model.state_dict(), f"mnist_{epoch}.pth")
