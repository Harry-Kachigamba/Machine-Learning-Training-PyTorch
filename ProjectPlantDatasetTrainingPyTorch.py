# LOADING THE DATASET PROCESS!

import os
from pickletools import optimize
from pyexpat import model
from turtle import forward
import torch
from torch.nn.modules import padding
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# Transformatioons of dataset
transformations = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Path to dataset directory
dataset_dir = "dataset location"

# Load the dataset
train_set = ImageFolder(root=os.path.join(dataset_dir, "train"), transform=transformations)
test_set = ImageFolder(root=os.path.join(dataset_dir, "test"), transform=transformations)

# Batch Size
batch_size = 10

# Data loaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=batch_size,  shuffle=False, num_workers=0)

# Printing details
print("The number of images in the training set is: ", len(train_loader) * batch_size)
print("The number of images in the test set is: ", len(test_loader) * batch_size)
print("The number of batches per epoch is: ", len(train_loader))
print("Classes:", train_set.classes)

# DEFINING A CONVOLUTIONAL NETWORK
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

# Defining the CNN
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24*10*10, 39)

def forward(self, input):
    output = F.relu(self.bn1(self.conv1(input)))
    output = F.relu(self.bn2(self.conv2(output)))
    output = self.pool(output)
    output = F.relu(self.bn4(self.conv4(output)))
    output = F.relu(self.bn5(self.conv5(output)))
    output = output.view(-1, 24*10*10)
    output = self.fc1(output)

    return output

# Creating an Instance of the neural network model with the definitions given
model = Network()

# DEFINING A LOSS
from torch.optim import adam

# Define the loss function with Classification Cross-Entropy loss and an Optimier with Adam Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# TRAINING THE MODEL ON THE AVAILABLE DATASET
from torch.autograd import Variable

# Function to save the model
def saveModel():
    path = "./ProjectPlantDiseaseModel.pth"
    torch.save(model.state_dict(), path())

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():

    model.eval()
    accuracy = 0.0
    total = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")

    with torch.mo_grad():
        for data in test_loader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images.to(device))
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels.to(device)).sum().item()

        # compute the accuracy over all test images
        accuraccy = (100 * accuracy / total)
        return(accuracy)


# Training function. Lopping over the data iterator and feeding the inputs to the netwrok and optimize
def train(num_epochs):

    best_accuracy = 0.0

    # Defining execution device (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Converting model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs): # Loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):

            # Getting the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # Zero the parameter gradients
            optimizer.zero_grad()
            # Predict classes using images from the training set 
            outputs = model(images)
            # Compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # Backpropagate the loss
            loss.backward()
            # Adjust paramaters based on the calculated gradients
            optimizer.step()

            # Printing statistics for every 100 images
            running_loss += loss.item() # Extract the loss value
            if i % 1000 == 999:
                # Print every 1000 (twice per epoch)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                # Zero the loss
                running_loss = 0.0

        # Compute an print the average accuracy tor this epoch when tested over 1000 test images
        accuracy = testAccuracy()
        print('For epoch', epoch+1, 'the test accuracy over the whole test set is %d %%' % (accuracy))

        # Saving the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy


# Testing the model on the test data
import matplotlib.pyplot as plt
import numpy as np

# Fucntions to show images
def imageShow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Fucntion to test the model with a batch of images and show the labels predictions
def testBatch():
    # Getting batch of images from the test DataLoader
    images, labels = next(iter(test_loader))

    # Show all images as one image grid
    imageShow(torchvision.utils.make_grid(images))

    #Show the real labels on the screen
    print('Real lables: ', ' '.join('%5s' % train_set.classes[labels[j]]
                                    for j in range(batch_size)))

    # Checking if the model identifies the labels of those examples
    outputs = model(images)

    # Probability for every 10 labels. The highest probability should be correct label
    _, predicted = torch.max(outputs, 1)

    # Displaying the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('&5s' %train_set.classes[predicted[j]]
                                  for j in range(batch_size)))

    # The main code to execute the whole thing
    if __name__ == "__main__":

        # Building the model
        train(10)
        print('Finished Training')

        #Testing classes with good performance
        testAccuracy()

        # Building the created model and test the accuracy per label
        model = Network()
        path = "ProjectPlantDiseaseModel.pth"
        model.load_state_dict(torch.load(path))

        # Test with batch of images
        testBatch()
