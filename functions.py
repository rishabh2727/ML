import torch
import torchvision
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

n_epochs = 5
batch_size_train = 200
batch_size_test = 1000
learning_rate = 1e-3
momentum = 0.5
log_interval = 100

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.nn.functional.softmax(x, dim = 1)
        return x

def train(epoch, data_loader, model, optimizer, criterion):
    model.train()  # Set the model to training mode
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)  
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))

def eval(data_loader, model, dataset, criterion):
    model.eval()  
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            loss += criterion(output, target).item()
    loss /= len(data_loader.dataset)
    print(dataset + 'set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, len(data_loader.dataset),
                                                                               100. * correct / len(data_loader.dataset)))
    accuracy = 100. * correct / len(data_loader.dataset)
    
    return accuracy,loss


def logistic_regression(dataset_name, device):
    if dataset_name == "MNIST":
        input_size = 28 * 28  
        num_classes = 10 
        training = torchvision.datasets.MNIST('./data/MNIST/', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

        test = torchvision.datasets.MNIST('./data/MNIST/', train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
    elif dataset_name == "CIFAR10":
        input_size = 32 * 32 * 3  
        num_classes = 10 
        transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        training = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        
        test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
        
    validation_size = 12000

    train_size = len(training) - validation_size
    training, validation = random_split(training, [train_size, validation_size])

    model = LogisticRegression(input_size, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-2)

    criterion = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(training, batch_size=batch_size_train, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation, batch_size=batch_size_train, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size_test, shuffle=True)

    for epoch in range(n_epochs):
        train(epoch, train_loader, model, optimizer, criterion)
        eval(validation_loader, model, "Validation", criterion)

    eval(test_loader, model, "Test", criterion)

    results = {
        "model": model, 
    }

    return results


# ____________________

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from sklearn.model_selection import ParameterGrid


def tune_hyper_parameter(dataset_name, target_metric, device):
    if dataset_name == "MNIST":
        input_size = 28 * 28
        num_classes = 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST('./data/MNIST/', train=True, download=True, transform=transform)
    elif dataset_name == "CIFAR10":
        input_size = 32 * 32 * 3
        num_classes = 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    else:
        raise ValueError("Invalid dataset name")

    validation_size = 12000
    train_size = len(dataset) - validation_size
    training_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    hyper_parameters = {
        'learning_rate': [0.001,0.01],
        'batch_size_train': [128],
        'n_epochs': [5,10], 
        'weight_decay':[1e-2],
        'optimizer': ['Adam', 'SGD'],
    }

    param_grid = list(ParameterGrid(hyper_parameters))

    best_params = None
    best_metric = float('-inf') if target_metric == 'acc' else float('inf')
    validation_loss = None
    validation_accuracy = 0.0
    

    for params in param_grid:
        if params['optimizer'] == 'Adam':
            optimizer = optim.Adam
        elif params['optimizer'] == 'SGD':
            optimizer = optim.SGD

        model = LogisticRegression(input_size, num_classes).to(device)
        optimizer = optimizer(model.parameters(), lr=params['learning_rate'],weight_decay=params["weight_decay"])

        train_loader = DataLoader(training_dataset, batch_size=params['batch_size_train'], shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=params['batch_size_train'], shuffle=True)

        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(1, params['n_epochs'] + 1):
            train(epoch, train_loader, model, optimizer, criterion) 
         
        validation_accuracy, validation_loss = eval(validation_loader, model, "Validation", criterion)
            

        if target_metric == 'acc' and validation_accuracy > best_metric:
            best_params = params
            best_metric = validation_accuracy
        elif target_metric == 'loss' and validation_loss is not None and validation_loss < best_metric:
            best_params = params
            best_metric = validation_loss

    return best_params, best_metric














