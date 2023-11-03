import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def get_data_loader(training=True):
    """
    TODO: implement this function.

    INPUT:
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Depending on the value of 'training', choose the train set or test set
    if training:
        train_set = datasets.FashionMNIST('./data', train=True, download=True, transform=custom_transform)
        loader = torch.utils.data.DataLoader(train_set, batch_size=64)

    else:
        train_set = datasets.FashionMNIST('./data', train=False, transform=custom_transform)
        loader = torch.utils.data.DataLoader(train_set, batch_size=64)

    return loader


def build_model():
    """
    TODO: implement this function.

    INPUT:
        None

    RETURNS:
        An untrained neural network model
    """

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128), # 28*28 = 784
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

    return model


def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT:
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy
        T - number of epochs for training

    RETURNS:
        None
    """
    # set up an optimizer
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train() # set model to train mode before iterating over the dataset

    # the outer for loop that iterates over epochs
    for epoch in range(T):
        epoch_loss = 0.0
        correct_predictions = 0
        # the inner for loop that iterates over batches of (images, labels) pairs from the train DataLoader
        for i, data in enumerate(train_loader, 0):
            images, labels = data
            opt.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(images)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
        # print the training status after every epoch of training
        print(f'Train Epoch: {epoch}\tAccuracy: {correct_predictions}/{len(train_loader.dataset)}({correct_predictions / len(train_loader.dataset) * 100:.2f}%)\tLoss: {epoch_loss / len(train_loader.dataset):.3f}')

# It prints the evaluation statistics as described below
# (displaying the loss metric value if and only if the optional parameter has not been set to False)
def evaluate_model(model, test_loader, criterion, show_loss=True):
    """
    TODO: implement this function.

    INPUT:
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy

    RETURNS:
        None
    """
    model.eval()
    correct_predictions = 0
    epoch_loss = 0.0

    with torch.no_grad(): # no need to track gradients during testing
        for data, labels in test_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            # Multiply by the actual batch size
            epoch_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
    if show_loss:
        print(f'Average loss: {epoch_loss / len(test_loader.dataset):.4f}')

    print(f'Accuracy: {correct_predictions / len(test_loader.dataset) * 100:.2f}%')


# It prints the top 3 most likely labels for the image at the given index, along with their probabilities
def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT:
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', "Ankle Boot"]
    # Predict only for the specific image at the given index
    output = model(test_images[index].unsqueeze(0))
    probs = F.softmax(output, dim=1)

    target_probs, labels = torch.topk(probs[0], 3)
    for i in range(3):
        print(f'{class_names[labels[i]]}: {target_probs[i] * 100:.2f}%')

if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()

    train_loader = get_data_loader()
    print(type(train_loader))
    #<class ’torch.utils.data.dataloader.DataLoader’>
    print(train_loader.dataset)
    '''
    Dataset FashionMNIST
        Number of datapoints: 60000
        Root location: ./data
        Split: Train
        StandardTransform
    Transform: Compose(
                 ToTensor()
                 Normalize(mean=(0.1307,), std=(0.3081,))
               )
    '''
    test_loader = get_data_loader(False)

    model = build_model()
    print(model)
    '''
    Sequential(
        (0): Flatten()
        (1): Linear(in_features=?, out_features=?, bias=True)
        (2): ReLU()
        (3): Linear(in_features=?, out_features=?, bias=True)
        ...
    )
    '''

    evaluate_model(model, test_loader, criterion, show_loss=False)
    #Accuracy: 85.39%
    evaluate_model(model, test_loader, criterion, show_loss=True)
    '''
    Average loss: 0.4116
    Accuracy: 85.39%
    '''
    predict_label(model, test_images, 1)
    '''
    Pullover: 92.48%
    Shirt: 5.93%
    Coat: 1.48%
    '''