# Rana AlHajri
# imports
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from PIL import Image

import argparse
parser = argparse.ArgumentParser(description='train file')
parser.add_argument('--data_dir', type=str, action="store", default="flowers", help='the directory of flower images')
parser.add_argument('--gpu', dest='gpu', action='store_true', default="gpu", help='activate the GPU during the training')
parser.add_argument('--save_dir', type=str,dest="save_dir", action="store", default="checkpoint.pth", help='directory to save checkpoints')
parser.add_argument('--arch', dest='arch', action="store", default="vgg16", type = str, help='model architecture')
parser.add_argument('--learning_rate', type=float, dest="learning_rate", action="store", default=0.001, help='learning rate')
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=10000, help='number of hidden units')
parser.add_argument('--epochs', type=int, dest="epochs", action="store", default=5, help='number of epochs')
parser.add_argument('--dropout', type=float, dest = "dropout", action = "store", default = 0.5, help='dropout percentage')

arg_parser = parser.parse_args()

data_dir = arg_parser.data_dir
gpu = arg_parser.gpu
save_dir = arg_parser.save_dir
model_arch = arg_parser.arch
lr = arg_parser.learning_rate
hidden_units = arg_parser.hidden_units
epochs = arg_parser.epochs
dropout = arg_parser.dropout

image_datasets_train = None

def load_data():
    #data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms_train = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    data_transforms_valid = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    data_transforms_test = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    image_datasets_train = datasets.ImageFolder(train_dir, transform=data_transforms_train)
    image_datasets_valid = datasets.ImageFolder(valid_dir, transform=data_transforms_valid)
    image_datasets_test = datasets.ImageFolder(test_dir, transform=data_transforms_test)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    #trainloader = torch.utils.data.DataLoader(image_datasets_train, batch_size=32, shuffle=True)
    #validloader = torch.utils.data.DataLoader(image_datasets_valid, batch_size=32)
    #testloader = torch.utils.data.DataLoader(image_datasets_test, batch_size=32)

    return image_datasets_train, image_datasets_valid, image_datasets_test

def model_architecture(lr=0.001, hidden_units=10000):
    # TODO: Build and train your network

    if model_arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_arch == 'vgg11':
        model = models.vgg11(pretrained=True)
    else:
        print("Please choose either VGG11 or VGG16")

    # Hyperparameters for our network
    input_size = 25088
    hidden_sizes = [10000, 500]
    output_size = 102

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(input_size, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(p=dropout),
                               nn.Linear(hidden_units, hidden_sizes[1]),
                               nn.ReLU(),
                               nn.Dropout(p=dropout),
                               nn.Linear(hidden_sizes[1], output_size),
                               nn.LogSoftmax(dim=1))

    model.classifier = classifier

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    return model, criterion, optimizer

# Implement a function for the validation pass
def validation(model, validloader, criterion):

    model.eval()
    valid_loss = 0
    accuracy = 0
    for images, labels in validloader:

        ### code reference ###
        # https://github.com/bryanfree66/AIPND_image_classification/blob/master/Image%20Classifier%20Project.ipynb
      
    if gpu == 'gpu':
        model.to('cuda')
    else:
        print("the model will be trained using gpu due to the performance")
        model.to('cuda')
        images = Variable(images.float().cuda(), volatile=True)
        labels = Variable(labels.long().cuda(), volatile=True)
        
        #####################

        output = model.forward(images) #%%
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy

def do_deep_learning(epochs=5):
  ##epochs = 5
    print_every = 40
    steps = 0

    if gpu == 'gpu':
        model.to('cuda')
    else:
        print("the model will be trained using gpu due to the performance")
        model.to('cuda')

    for e in range(epochs):
        model.train() #

        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')
  

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:            
                model.eval()
                
                #  validation
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion)
                
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                
                running_loss = 0
                
                model.train()

def save_checkpoint(model):
    # TODO: Save the checkpoint
    model.class_to_idx = image_datasets_train.class_to_idx

    checkpoint = {'model_arch': model_arch,
                  'hidden_units':hidden_units,
                  'learning_rate': lr,
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'optimizer_dict':optimizer.state_dict(),
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, save_dir)
if __name__== "__main__":

    print ("the training begin")
    image_datasets_train, image_datasets_valid, image_datasets_test = load_data()
    print ("loading part done")
    model, criterion, optimizer = model_architecture(lr, hidden_units)
    trainloader = torch.utils.data.DataLoader(image_datasets_train, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(image_datasets_valid, batch_size=32)
    testloader = torch.utils.data.DataLoader(image_datasets_test, batch_size=32)
    model, criterion, optimizer = model_architecture(lr, hidden_units)
    do_deep_learning(epochs)
    print ("learning part done")
    
    save_checkpoint(model)
    print ("the end of the training")
