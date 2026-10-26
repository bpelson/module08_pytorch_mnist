import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


# %% Command line arguments
def get_arguments():
    parser = argparse.ArgumentParser(description='MNIST Classification with PyTorch')
    parser.add_argument('-e', '--num_epochs', type=int, default=2,
                        help='Number of training epochs (default: 2)')
    parser.add_argument('-b', '--batch_size', type=int, default=1000,
                        help='Number of samples per batch (default: 1000)')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    return parser.parse_args()


# %% Convolutional neural network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # First convolutional layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Second convolutional layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Fully connected layers
        self.fc1 = nn.Linear(7*7*64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        # Forward pass through convolutional layers
        out = self.layer1(x)
        out = self.layer2(out)

        # Flatten output and pass through connected layers
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# %% Track accuracy
def track_accuracy(model, loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
    model.train()
    return accuracy


# %% Training
def training(model, num_epochs, train_loader, test_loader, criterion,
             optimizer, device):
    # Store loss and accuracy values
    loss_list = []
    acc_train = []
    acc_test = []

    for epoch in range(num_epochs):
        pbar = tqdm(train_loader,
                    desc=f'Training: {epoch + 1:03d}/{num_epochs}',
                    ncols=125,
                    leave=True)

        running_loss = []

        for i, (images, labels) in enumerate(pbar, 1):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass and loss
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss.append(loss.item())

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f'{sum(running_loss)/i : 10.6f}')

        # Store batch loss for this epoch
        loss_list.append(sum(running_loss)/i)

        # Track train and test set accuracy
        acc_train.append(track_accuracy(model, train_loader, device))
        acc_test.append(track_accuracy(model, test_loader, device))

    return model, loss_list, acc_train, acc_test


# %% Plot
def plot(loss_list, acc_train, acc_test, save_path='./figure.png'):
    # Plot loss and accuracy curves
    plt.figure(figsize=(10, 4))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    # Plot train and test accuracies over epochs
    plt.subplot(1, 2, 2)
    plt.plot(acc_train)
    plt.plot(acc_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend(['Training', 'Testing'])

    # Save figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# %% Main run
def main(num_epochs, batch_size, learning_rate):
    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    # Create model and move it to device
    model = ConvNet().to(device)

    # Loss/optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    model, loss_list, acc_train, acc_test = training(
        model, num_epochs, train_loader, test_loader, criterion, optimizer, device)

    # Plot the results
    plot(loss_list, acc_train, acc_test)