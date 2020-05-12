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
import json

import argparse
parser = argparse.ArgumentParser(description='predict file')
parser.add_argument('--input', default='./flowers/test/1/image_06743.jpg', action="store", type = str, help='image path')
parser.add_argument('--checkpoint', default='./checkpoint.pth', action="store", type = str, help='checkpoint file path/name')
parser.add_argument('--top_k', default= 5, dest="top_k", action="store", type=int, help='return top k classes ')
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json', help='mapping the categories to real names')
parser.add_argument('--gpu', dest='gpu', action='store_true', default="gpu", help='GPU is activated during predection ')
#parser.add_argument('--arch', dest='arch', action='store', default="vgg16", type = str, help='model arch ')

arg_parser = parser.parse_args()
image_path =arg_parser.input
model_path = arg_parser.checkpoint
topk = arg_parser.top_k
gpu = arg_parser.gpu
category_names = arg_parser.category_names
# load_checkpoint()
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['model_arch'] == 'vgg16':
        model2 =models.vgg16(pretrained=True)
    for param in model2.parameters():
        param.requires_grad = False
    #model2.classifier = checkpoint['classifier']
    
    # hyperparameters
    input_size = 25088
    hidden_sizes = [10000, 500]
    output_size = 102
    
    classifier = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(hidden_sizes[1], output_size),
                               nn.LogSoftmax(dim=1))
    model2.classifier = classifier 
                           
    
    model2.class_to_idx = checkpoint['class_to_idx']
    model2.opitimizer_dict = checkpoint['optimizer_dict']
    model2.load_state_dict(checkpoint['state_dict'])
    lr = checkpoint['learning_rate']
    epoch = checkpoint['epochs']
    
    return model2

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image)
    
    image_preprocessing = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.244, 0.225])])
    image = image_preprocessing(image)
    return image.numpy()
    
#img = process_image('flowers/test/1/image_06743.jpg')
#print(img.shape)



def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    model.to('cuda')
    model.eval()
    np_array = process_image(image_path)
    image = torch.from_numpy(np_array)
    image = Variable(image.float().cuda(), volatile=True)
    image = image.unsqueeze(0)
    
    output = model.forward(image)
    
    ps = torch.exp(output)
    
    return ps.topk(topk)




def view_classify(img):
    ''' Function for viewing an image and it's predicted classes.
    '''
   ## ps = ps.data.numpy().squeeze()
    probs, indexs = predict(img, model)
    probs = probs.data.cpu().numpy()[0]
    indexs = indexs.data.cpu().numpy()[0]
    ## ADD MAPPING
    #x_prob = []
    #y_index = []
    y_class = []
    y_label = []
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    for i in indexs:
        y_class.append(i)
        y_label.append(cat_to_name[str(i + 1)])
    
    # Convert indices to classes
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [index_to_class[each] for each in y_class]
 

    
    ##
    print ('x_prob: ', probs, 'top_indexs: ', indexs)

    flower_by_class = []
    for i in top_classes:
        flower_by_class.append(cat_to_name[i])
    
    print ('y_by_class: ',flower_by_class)

if __name__== "__main__":
    print("predection process started")
    model = load_checkpoint(model_path)
    #probs, indexs = predict(image_path, model, topk)
    view_classify(image_path)
    print("predection process done")

