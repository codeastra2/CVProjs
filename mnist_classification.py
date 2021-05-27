import torch
import torchvision
import torchvision.datasets as datasets
from numpy.core import argmax, vstack
from sklearn.metrics import accuracy_score
from torch.nn import Conv2d, ReLU, MaxPool2d, Linear, Softmax, CrossEntropyLoss
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from matplotlib import pyplot





'''#Plotting some images
for i in range(25):
    # Defining the plot
    pyplot.subplot(5, 5, i+1)

    pyplot.imshow(inputs[i][0], cmap='gray')
    #print(inputs[i][0].shape)


#Displaying the image
print(len(train_dl))
pyplot.show()'''

class CNN(torch.nn.Module):
    """
    Class defining the CNN model.
    """

    #Defining the model
    def __init__(self, n_channels):
        """

        """

        super(CNN, self).__init__()

        #Defnining the first hidden layer
        self.hidden_layer1 = Conv2d(n_channels, 32, (3, 3))
        kaiming_uniform_(self.hidden_layer1.weight, nonlinearity="relu")
        #First layer activation function
        self.act1 = ReLU()

        #Defining the first pooling layer
        self.pool1 = MaxPool2d((2, 2), stride=2)

        #Defning he second hidden layer
        self.hidden_layer2 = Conv2d(32, 32, (3, 3))
        kaiming_uniform_(self.hidden_layer2.weight, nonlinearity="relu")
        # Second layer activation function
        self.act2 = ReLU()

        #Second pooling layer
        self.pool2 = MaxPool2d((2, 2), stride=2)

        #Fully connected layer
        self.hidden_layer3 = Linear(5*5*32, 100)
        kaiming_uniform_(self.hidden_layer3.weight, nonlinearity="relu")
        # Third layer activation function
        self.act3 = ReLU()

        #Output layer
        self.hidden_layer4 = Linear(100, 10)
        xavier_uniform_(self.hidden_layer4.weight)
        self.act4 = Softmax(dim=1)


    # Forward propagation of model
    def forward(self, X):
        # Apply the first hidden layer
        X = self.hidden_layer1(X)

        # Applying the first activation function
        X = self.act1(X)

        #Applying the first pooling layer
        X = self.pool1(X)

        #Applying the second hidden CNN layer
        X = self.hidden_layer2(X)

        # Applying the second activation function
        X = self.act2(X)

        #APplying the second pooling function
        X = self.pool2(X)

        #Changing dimensions
        X = X.view(-1,4*4*50)

        # Third hidden layer
        X = self.hidden_layer3(X)

        #Applying the activation function
        X = self.act3(X)

        #Applying the output layer
        X = self.act4(X)
        X = self.act4(X)

        return X


def train_model(train_dl, model):
    # Define the optimization

    #Optimization criterion
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    #Enumerate epochs
    for epoch in range(10):


        #Enumerating the mini batches
        for i, (inputs, targets) in enumerate(train_dl):

            print("Processing the epoch number " + str(epoch) + " and batch number: " + str(i))
            #Clear the gradients
            grads = optimizer.zero_grad()

            #COmpute the predictions
            y_hat = model(inputs)

            #Compute the loss
            loss = criterion(y_hat, targets)

            # Accumulate the gradients
            loss.backward()

            #Update the weights of the model
            optimizer.step()


def prepare_dataset():
    trans = Compose([ToTensor()])

    # The datasets
    mnist_trainset = datasets.MNIST(root="./datasets", train=True, download=True, transform=trans)
    mnist_testset = datasets.MNIST(root="./datasets", train=False, download=True, transform=trans)

    # Defining the dataloaders

    train_dl = DataLoader(mnist_trainset, batch_size=32, shuffle=True)
    test_dl = DataLoader(mnist_testset, batch_size=32, shuffle=True)

    return train_dl, test_dl

def evaluate_model(test_dl, model):

    predictions, actuals = list(), list()

    #Going oer the testset minibatches

    for i, (inputs, targets) in enumerate(test_dl):

        # Evaluate the model on the test set
        y_hat = model(inputs)

        #Retreive the numpy array
        y_hat = y_hat.detach().numpy()
        actual = targets.numpy()

        #Convert to class labels
        y_hat = argmax(y_hat, axis=1)

        # Reshape for stacking
        actual = actual.reshape((len(actual), 1))
        y_hat = y_hat.reshape((len(y_hat), 1))

        predictions.append(y_hat)
        actuals.append(actual)

    predictions , actuals = vstack(predictions), vstack(actuals)
    acc = accuracy_score(actuals, predictions)
    return acc

train_dl, test_dl = prepare_dataset()
print(len(train_dl.dataset), len(test_dl.dataset))

model = CNN(1)

train_model(train_dl, model)

acc = evaluate_model(test_dl, model)

print("The accuracy is: " + str(acc))











