"""
Describes the model architecture and utility functions.
"""


from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets


# Define the model architecture
class Net(nn.Module):
    """
    2-layer fully-connected neural network
    with dropout and ReLU activation functions.
    784 -> 512 -> 512 -> 10 dimensions.
    """
    def __init__(self) -> None:
        super(Net, self).__init__()

        # linear layer 784 -> 512
        self.fc1 = nn.Linear(784, 512)
        # linear layer 512 -> 512
        self.fc2 = nn.Linear(512, 512)
        # linear layer 512 -> 10
        self.fc3 = nn.Linear(512, 10)

        # dropout layer
        self.dropout = nn.Dropout(0.2)

        # activation function
        self.activation = nn.ReLU()

        self.layers = nn.Sequential(
            *[
                self.fc1,
                self.activation,
                self.dropout,
                self.fc2,
                self.activation,
                self.dropout,
                self.fc3,
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten input
        x = x.view(-1, 784)
        return self.layers(x)

    def inference(self, x: torch.Tensor) -> Tuple[int, torch.Tensor]:
        output = self.forward(x)
        _, y_hat = torch.max(output, 1)
        return y_hat, output


def train(
    model: nn.Module, loss_function: nn.Module, optimizer: Optimizer, n_epochs: int = 50
) -> None:
    valid_loss_min = np.Inf

    # Run through the dataset ``n_epoch`` times
    for epoch in range(n_epochs):
        train_loss = 0
        valid_loss = 0

        # Put the model in training mode (keep track of gradients)
        model.train()
        for data, label in train_loader:

            # set the gradients to zero
            optimizer.zero_grad()

            # forward pass through the model
            output = model(data)

            # compute the loss
            loss = loss_function(output, label)

            # calculate gradients
            loss.backward()

            # update weights
            optimizer.step()

            # update training loss tracker
            train_loss += loss.item() * data.size(0)

        # At the end of each epoch, validate
        model.eval()
        for data, label in valid_loader:
            output = model(data)
            loss = loss_function(output, label)
            valid_loss += loss.item() * data.size(0)

        train_loss /= len(train_loader.sampler)
        valid_loss /= len(valid_loader.sampler)

        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                epoch + 1, train_loss, valid_loss
            )
        )

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print(
                "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                    valid_loss_min, valid_loss
                )
            )
            torch.save(model.state_dict(), "model.pt")
            valid_loss_min = valid_loss


def test(model: nn.Module, loss_function: nn.Module) -> None:
    test_loss = 0
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))

    model.eval()
    for data, target in test_loader:
        y_hat, output = model.inference(data)

        # compute and save the loss
        loss = loss_function(output, target)
        test_loss += loss.item() * data.size(0)

        # compare predictions to label
        correct = np.squeeze(y_hat.eq(target.data.view_as(y_hat)))
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print average test loss
    test_loss /= len(test_loader.sampler)
    print("Test loss: {:.6f}\n".format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print(
                "Test Accuracy of %5s: %2d%% (%2d/%2d)"
                % (
                    str(i),
                    100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]),
                    np.sum(class_total[i]),
                )
            )
        else:
            print("Test Accuracy of %5s: N/A (no training examples)" % (str(i)))

    print(
        "\nTest Accuracy (Overall): %2d%% (%2d/%2d)"
        % (
            100.0 * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct),
            np.sum(class_total),
        )
    )

if __name__ == "__main__":

    ## Preamble
    num_workers = 0
    batch_size = 20
    valid_size = 0.2

    ## Data Loading
    transform = transforms.ToTensor()

    # Training and testing datasets
    train_data = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    # Get training indices for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_index, valid_index = indices[split:], indices[:split]

    # Define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)

    # Prepare data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, sampler=None, num_workers=num_workers
    )

    ## Create the model
    model = Net()
    print(model)

    ## Loss funtion
    loss = nn.CrossEntropyLoss()

    ## Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    ## Train the model
    train(model=model, loss_function=loss, optimizer=optimizer, n_epochs=50)

    ## Test the model
    test(model=model, loss_function=loss)
