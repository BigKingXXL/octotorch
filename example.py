import os
import timm
import torch
import torchvision
from octotorch import OctoTorch
from tqdm import tqdm

mnist_data = torchvision.datasets.MNIST('./MNIST', download=True, transform=torchvision.transforms.ToTensor())
data_loader = torch.utils.data.DataLoader([mnist_data[x] for x in range(1000)],
                                          batch_size=100,
                                          shuffle=True)

def train_once(model: torch.nn.Module):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    for _ in range(5):
        for x,y in tqdm(data_loader):
            optimizer.zero_grad()
            y_hat = model(x)
            l = torch.functional.F.cross_entropy(y_hat, y)
            l.backward()
            optimizer.step()

def score(model: torch.nn.Module):
    model.eval()
    n_correct = 0
    for x,y in tqdm(data_loader, leave=False):
        y_hat = model(x).argmax(dim=1)
        n_correct += sum(y == y_hat)
    return (n_correct / 1000).item()

model = torch.nn.Sequential(torch.nn.Conv2d(1,3,1), timm.create_model("resnet26", pretrained=True, num_classes=10))

if os.path.exists('./MNIST/pretrained.pth'):
    model.load_state_dict(torch.load('./MNIST/pretrained.pth'))
else:
    train_once(model)
    torch.save(model.state_dict(), './MNIST/pretrained.pth')

data_loader = torch.utils.data.DataLoader([mnist_data[x] for x in range(100)],
                                          batch_size=20,
                                          shuffle=True)

print("score without everything: ", score(model))
OctoTorch(model, score_func=score, allow_layers=["conv"]).quantize()