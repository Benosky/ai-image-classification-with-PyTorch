
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
import json


# Initialize the parser
parser = argparse.ArgumentParser(
    description="Benjamin_Umeh Flower Classification Deep Neural Network"
    )

# Add the parameters positional
parser.add_argument('path_to_image', help="path to the image you want to classify.")
parser.add_argument('checkpoint', help="Name of the checkpoint saved during training. Ends with .pth extention. Eg checkpoint.pth")

# Add the parameters optional
parser.add_argument('-t','--topk', help="Top k most likely classes", type=int, default=5)
parser.add_argument('-c','--cat_names', help="The .json file of the mapping of categories to real names", default='cat_to_name.json')
parser.add_argument('-g','--gpu', help="Choose to use gpu if available; returns to cpu is not available: enter y or n", default='n')

# Parse the arguments
args = parser.parse_args()

#Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == "densenet":
            model_ft = models.densenet121(pretrained=True)
    elif checkpoint['arch'] == "vgg11_bn":
        model_ft = models.vgg11_bn(pretrained=True)
    elif checkpoint['arch'] == "vgg13":
        model_ft = models.vgg13(pretrained=True)
    elif checkpoint['arch'] == "vgg16":
        model_ft = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == "vgg19":
        model_ft = models.vgg19(pretrained=True)
    elif checkpoint['arch'] == "alexnet":
        model_ft = models.alexnet(pretrained=True)
    model_ft.classifier = nn.Sequential(nn.Linear(checkpoint['input_size'], checkpoint['hidden_layer_size']),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(checkpoint['hidden_layer_size'], checkpoint['output_size']),
                        nn.LogSoftmax(dim=1))
    model_ft.load_state_dict(checkpoint['state_dict'])
    model_ft.class_to_idx = checkpoint["class_to_idx"]
    return model_ft


model = load_checkpoint(args.checkpoint)
classifier_model = model.classifier

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    with Image.open(image) as im:
        img = im.copy()
        image_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
        img = image_transforms(img)#.cpu().detach().numpy()
        return img

sample_image = process_image(args.path_to_image)

# Use GPU if it's available and is chosen
is_chosen = args.gpu.upper()
device = torch.device("cuda" if torch.cuda.is_available() and is_chosen == "Y" else "cpu")


with open(args.cat_names, 'r') as f:
    cat_to_name = json.load(f)


def predict(image_path, model, topk=args.topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    # Calculate the class probabilities (softmax) for img
    cuda = torch.cuda.is_available() 
    if cuda:
        model = model.cuda()
    else:
        model = model.cpu()
    with torch.no_grad():
        tensor = process_image(image_path)
        if cuda:
            inputs = tensor.float().cuda()
        else: 
            inputs = tensor
        inputs = inputs.unsqueeze(0)
        output = model.forward(inputs)        
    ps = torch.exp(output)
    print("Shape of ps is: ", ps.shape)
    top_p, top_class = ps.topk(topk)
    inverted_class_to_idx = {model.class_to_idx[k]: k for k in model.class_to_idx}
    classes_map = list()
    for label in top_class.cpu().numpy()[0]:
        classes_map.append(inverted_class_to_idx[label])
    return top_p.cpu().numpy()[0], classes_map


model.to(device)

probs, classes = predict(args.path_to_image, model)
flower_names = [cat_to_name[x] for x in classes]

print(classes)
print(flower_names)
print(probs)
