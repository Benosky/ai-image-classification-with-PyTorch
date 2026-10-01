# Imports here

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from workspace_utils import active_session
from torch import optim
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import pandas as pd
import argparse


# Initialize the parser
parser = argparse.ArgumentParser(
    description="Benjamin_Umeh Flower Classification Deep Neural Network"
    )

# Add the parameters positional
parser.add_argument('data_dir', help="The Directory in torchvision.datasets.ImageFolder where our data is located. Eg flowers")
parser.add_argument('checkpoint', help="Set directory to save checkpoints. Ends with .pth extention. Eg checkpoint.pth")

# Add the parameters optional
parser.add_argument('-a','--arch', help="Choose the pretrained model achitecture from torchvision.models to get the \
                        image features. List to choose from: , densenet, alexnet, vgg, alexnet,\
                         etc", default='densenet')
parser.add_argument('-l','--learning_rate', help="Model learning rate", type=float, default=0.003)
parser.add_argument('-u','--hidden_units', help="Number of units in the deep/hidden layers", type=int, default=900)
parser.add_argument('-e','--epochs', help="Model epochs", type=int, default=9)
parser.add_argument('-d','--device', help="Choose to use which device to use (cpu or gpu); returns to cpu if gpu is not available", default='gpu')
parser.add_argument('-f','--loss', help="Choose the loss function to use", default='NLL')
parser.add_argument('-b','--batch_size', help="Choose the batch size to use", type=int, default=32)
# Parse the arguments
args = parser.parse_args()
if (args.arch=='help'):
    print("List of available CNN networks:-")
    print("1. vgg11_bn (default)")
    print("2. vgg13")
    print("3. vgg16")
    print("4. vgg19")
    print("5. densenet121")
    print("6. alexnet")
    quit()

if (args.loss=='help'):
    print("List of available loss functions:-")
    print("1. L1 - Mean absolute error")
    print("2. NLL - Negative Log Likelihood loss (default)")
    print("3. Poisson - NLL loss with poisson distribution")
    print("4. MSE - Mean Squared Error loss")
    print("5. Cross - CrossEntropyLoss")
    quit()

if (not(args.learning_rate>0 and args.learning_rate<1)):
    print("Error: Invalid learning rate")
    print("Must be between 0 and 1 exclusive")
    quit()

if (args.batch_size<=0):
    print("Error: Invalid batch size")
    print("Must be greater than 0")
    quit()

if (args.epochs<=0):
    print("Error: Invalid epoch value")
    print("Must be greater than 0")
    quit()

if (args.hidden_units<=0):
    print("Error: Invalid number of hidden units given")
    print("Must be greater than 0")
    quit()


arches = ["vgg11_bn", "vgg13", "vgg16", "vgg19", "alexnet", "densenet"]
lossF = ["L1", "NLL", "Poisson", "MSE", "Cross"]

if args.arch not in arches:
    print("Error: Invalid architecture name received")
    print("Type \ 'python train.py -a help\' for more information")
    quit()


if args.loss not in lossF:
    print("Error: Invalid loss function name received")
    print("Type \ 'python train.py -f help\' for more information")
    quit()


if args.device not in ['cpu', 'gpu']:
    print("Error: Invalid device name received")
    print("It must be either 'cpu' or 'gpu'")
    quit()


#Load the data
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define your transforms for the training, validation, and testing sets


train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
"""

How PyTorch Reads Image Numbers
When PyTorch reads a colored image, it breaks it down into a 3D grid of numbers called a Tensor with the shape 
structure [Channels, Height, Width]:

1. Channels (C): Usually 3 for standard color images (Red, Green, Blue) or 1 for black and white (grayscale).
2. Height (H) & Width (W): The pixel resolution of your image.
3. The Value Range: By default, using to_tensor() scales the raw pixel values down from integers between 0 and 255 
   into floating-point numbers between 0.0 and 1.0. 0.0 represents pure black (no color), and 1.0 represents pure 
   white (maximum intensity).

Batching for Machine Learning
When feeding multiple images into a neural network, you typically group them into batches. This adds a fourth 
dimension to the front, changing the shape to [Batch Size, Channels, Height, Width].



--------------------------------------------------------------------------------------------------------------
transforms.Normalize([0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225])

This line of code normalizes an image tensor using the specific mean and standard deviation of the 
ImageNet dataset. It scales your input data so it matches the distribution of the images used to pre-train 
popular computer vision models.

📊 Parameter Breakdown
1. First List [0.485, 0.456, 0.406]: Mean for Red, Green, and Blue channels.
2. Second List [0.229, 0.224, 0.225]: Standard deviation for Red, Green, and Blue 
channels.

⚙️ How It Works
1. Expects a float tensor input.
2. Expects values between 0.0 and 1.0.
3. Formula applied: output[channel] = (input[channel] - mean[channel]) / std[channel].

"""

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)



# TODO: Using the image datasets and the trainforms, define the dataloaders

trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
validloader = torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size)
testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)

print("data loaded successfully")

# Building and training the classifier

# Use GPU if it's available and is chosen else use cpu
if args.device == "gpu":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# Choose the pre-trained model to use
def initialize_model(model_arch, num_of_hidden_units = None):
    model_ft = None

    #Set values for hidden hidden_units
    arch_to_hidden_units = {"vgg13": 512, "vgg16": 512, "vgg11_bn": 512, "vgg19": 512,
    "alexnet": 512, "densenet121": 512}

    # Different architectures
    if model_arch == "densenet":
        model_ft = models.densenet121(pretrained=True)
    elif model_arch == "vgg11_bn":
        model_ft = models.vgg11_bn(pretrained=True)
    elif model_arch == "vgg13":
        model_ft = models.vgg13(pretrained=True)
    elif model_arch == "vgg16":
        model_ft = models.vgg16(pretrained=True)
    elif model_arch == "vgg19":
        model_ft = models.vgg19(pretrained=True)
    elif model_arch == "alexnet":
        model_ft = models.alexnet(pretrained=True)
    else:
        print("Selected wrong or unavailable model architecture")

    if model_ft is not None:
        # Define hidden layer batch_size
        if num_of_hidden_units is None:
            num_of_hidden_units = arch_to_hidden_units[model_arch]

        # Freeze parameters so we don't backprop through them
        for param in model_ft.parameters():
            param.requires_grad = False

        # number of input features for classifier
        model_last_child = list(model_ft.children())[-1]
        if (isinstance(model_last_child, nn.modules.linear.Linear)):
            in_features = model_last_child.in_features
        else:
            list_of_children = list(model_last_child.children())
            for i in range(len(list_of_children)):
                if (isinstance(list_of_children[i], nn.modules.linear.Linear)):
                    in_features = list_of_children[i].in_features
                    break

        # Create the network
        classifier = nn.Sequential(nn.Linear(in_features, num_of_hidden_units),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(num_of_hidden_units, 102),
                              nn.LogSoftmax(dim=1))
        
        if model_arch in ["alexnet","densenet"]:
            model_ft.classifier = classifier
        else:
            model_ft = classifier        

    return model_ft, in_features 

print("Network created successfully")

## Define the criterion and optimizer
#lossF = ["L1", "NLL", "Poisson", "MSE", "Cross"])
if args.loss == "NLL":
    criterion = nn.NLLLoss()
elif args.loss == "L1":
    criterion = nn.L1Loss()
elif args.loss == "Poisson":
    criterion = nn.PoissonNLLLoss()
elif args.loss == "MSE":
    criterion = nn.MSELoss()
elif args.loss == "Cross":
    criterion = nn.CrossEntropyLoss()


model_ft, input_size = initialize_model(args.arch, num_of_hidden_units = args.hidden_units)


# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model_ft.classifier.parameters(), lr=args.learning_rate)

model_ft.to(device)

# Train the model and do validation on the test set
with active_session():

    epochs = args.epochs
    steps = 0

    train_losses, validation_losses = [], []
    train_accuracies, validation_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(epochs):
        running_loss = 0
        run_accuracy = 0
        for inputs, labels in trainloader:
    #         steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            logps = model_ft.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            running_ps = torch.exp(logps)
            run_top_p, run_top_class = running_ps.topk(1, dim=1)
            run_equals = run_top_class == labels.view(*run_top_class.shape)
            run_accuracy += torch.mean(run_equals.type(torch.FloatTensor)).item()

        else:
            validation_loss = 0
            accuracy = 0
            testing_loss = 0
            testing_accuracy = 0
            with torch.no_grad():
                # set model to evaluation mode
                model_ft.eval()
                for images, labels in validloader:
                    #Move input and label tensors to the default device
                    images, labels = images.to(device), labels.to(device)
                    # Get the class probabilities
                    log_ps = model_ft.forward(images)
                    batch_loss = criterion(log_ps, labels)
                    validation_loss += batch_loss.item()
    #                 test_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            # set model back to train mode
            model_ft.train() 
            train_losses.append(running_loss/len(trainloader))
            validation_losses.append(validation_loss/len(validloader))
            train_accuracies.append(run_accuracy/len(trainloader))
            validation_accuracies.append(accuracy/len(validloader))
            print("Epochs: {}/{}...".format(epoch+1, epochs),
                  "Training Loss: {:.3f}".format(running_loss/len(trainloader)),
                  "Training Accuracy: {:.3f}".format(run_accuracy/len(trainloader)),
                  "Validation Loss: {:.3f}".format(validation_loss/len(validloader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
            with torch.no_grad():
                # set model to evaluation mode
                model_ft.eval()
                for images, labels in testloader:
                    #Move input and label tensors to the default device
                    images, labels = images.to(device), labels.to(device)
                    # Get the class probabilities
                    test_log_ps = model_ft.forward(images)
                    test_batch_loss = criterion(test_log_ps, labels)
                    testing_loss += test_batch_loss.item()
            #                 test_loss += criterion(log_ps, labels)

                    test_ps = torch.exp(test_log_ps)
                    test_top_p, test_top_class = test_ps.topk(1, dim=1)
                    equals = test_top_class == labels.view(*test_top_class.shape)
                    testing_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            # set model back to train mode
            model_ft.train() 
            test_losses.append(testing_loss/len(testloader))
            test_accuracies.append(testing_accuracy/len(testloader))
            print("Epochs: {}/{}...".format(epoch+1, epochs),
                  "Testing Loss: {:.3f}".format(testing_loss/len(testloader)),
                  "Testing Accuracy: {:.3f}".format(testing_accuracy/len(testloader)))
            
            
            
#             test_loss = 0
#             accuracy = 0
#             with torch.no_grad():
#                 # set model to evaluation mode
#                 model_ft.eval()
#                 for images, labels in validloader:
#                     #Move input and label tensors to the default device
#                     images, labels = images.to(device), labels.to(device)
#                     # Get the class probabilities
#                     log_ps = model_ft.forward(images)
#                     batch_loss = criterion(log_ps, labels)
#                     test_loss += batch_loss.item()
#     #                 test_loss += criterion(log_ps, labels)

#                     ps = torch.exp(log_ps)
#                     top_p, top_class = ps.topk(1, dim=1)
#                     equals = top_class == labels.view(*top_class.shape)
#                     accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
#             # set model back to train mode
#             model_ft.train()
#             train_losses.append(running_loss/len(trainloader))
#             test_losses.append(test_loss/len(validloader))
#             print("Epochs: {}/{}...".format(epoch+1, epochs),
#                   "Training Loss: {:.3f}".format(running_loss/len(trainloader)),
#                   "Validation Loss: {:.3f}".format(test_loss/len(validloader)),
#                   "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
#             with torch.no_grad():
#                 # set model to evaluation mode
#                 model.eval()
#                 for images, labels in testloader:
#                     #Move input and label tensors to the default device
#                     images, labels = images.to(device), labels.to(device)
#                     # Get the class probabilities
#                     test_log_ps = model.forward(images)
#                     test_batch_loss = criterion(test_log_ps, labels)
#                     testing_loss += test_batch_loss.item()
#             #                 test_loss += criterion(log_ps, labels)

#                     test_ps = torch.exp(test_log_ps)
#                     test_top_p, test_top_class = test_ps.topk(1, dim=1)
#                     equals = test_top_class == labels.view(*test_top_class.shape)
#                     testing_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
#             # set model back to train mode
#             model.train() 
#             test_losses.append(testing_loss/len(testloader))
#             test_accuracies.append(testing_accuracy/len(testloader))
#             print("Epochs: {}/{}...".format(epoch+1, epochs),
#                   "Testing Loss: {:.3f}".format(testing_loss/len(testloader)),
#                   "Testing Accuracy: {:.3f}".format(testing_accuracy/len(testloader)))
            
            
            
            
# Save mapping of classes to indices
model_ft.class_to_idx = train_data.class_to_idx

#Save the checkpoint

checkpoint = {
              'input_size': input_size, 
              'hidden_layer_size': args.hidden_units,
              'output_size': 102,
              'arch' : args.arch,
#               'classifier_state_dict': model_ft.classifier.state_dict(),
              'class_to_idx': model_ft.class_to_idx,
              'optimizer_state': optimizer.state_dict(),
              'state_dict': model_ft.state_dict()
             }

torch.save(checkpoint, args.checkpoint)
