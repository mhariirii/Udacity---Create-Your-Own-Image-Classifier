import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision .models as models
from torch.autograd import variable
from collections import OrderedDict
from PIL import Image
import json
import utilities


def model_set(arch, learning_rate, hidden_units):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if arch == 'vgg19':
        model = models.vgg19(pretrained = True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif arch == 'vgg13':
        model = models.vg13(pretrained = True)
    else:
        print("{} vgg model is not a valid model to choose".format(arch))
        
        
    for param in model. parameters():
        param.requires_grad = False
        
    model.classifier= nn.Sequential(OrderedDict([
                             ('fc1', nn.Linear(25088, hidden_units)),
                             ('relu', nn.ReLU()),
                             ('drop', nn.Dropout(p=0.5)),
                             ('fc2', nn.Linear(hidden_units, 102)),
                             ('output', nn.LogSoftmax(dim=1))
                             ]))
                      
    criterion = nn.NLLLoss()
                     
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.002) 
    
    model.to(device)
                     
    return model, criterion, optimizer     

def train_model(model, trainloader, validloader, epoch, criterion, optimizer):
    
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                     
     epochs = epoch
     steps = 0
     running_loss = 0
     print_every = 50
                      
     
     for e in range(epochs):
         for inputs, labels in trainloader:
             steps += 1
             inputs, labels = inputs.to(device), labels.to(device)
             
             optimizer.zero_grad()
             
             logps = model.forward(inputs)
             loss = criterion(logps, labels)
             loss.backward()
             optimizer.step()
             
             running_loss += loss.item()
        
             if steps % print_every == 0:
                 test_loss = 0
                 accuracy = 0
                 model.eval()
                 
                 with torch.no_grad():
                     for inputs, labels in validloader:
                         inputs, labels = inputs.to(device), labels.to(device)
                         logps = model.forward(inputs)
                         
                         loss1 = criterion(logps, labels)
                          
                         test_loss += loss1.item()
                         
                         ps = torch.exp(logps)
                         top_p, top_class = ps.topk(1, dim=1)
                         equals = top_class == labels.view(*top_class.shape)
                         accuracy += torch.mean(equals.type(torch.FloatTensor))
                
                 print("Epoch: {}/{}..".format(e+1, epochs),
                       f"Train Loss: {running_loss/len(trainloader):.3f}.. "
                       f"validation Loss: {test_loss/len(validloader):.3f}.." 
                       f"Validation Accuracy: {accuracy/len(validloader):.3f}")
                 running_loss = 0 
                 model.train() 
                      
def saving_checkpoint(model, save_dir, hidden_units, optimizer, train_data):
    
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'classifier': model.classifier,
                  'arch': 'vgg19',
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, 'checkpoint.pth')

def load_checkpoint(filepath):
    checkpoints = torch.load(filepath)
    model=models.vgg19(pretrained=True)
    model.class_to_idx = checkpoints['class_to_idx']
    
    model.classifier = checkpoints['classifier']
    
    model.load_state_dict(checkpoints['state_dict'])
    
    
    
    return model


                      
                      
def process_image(image):
    
    img_pl = Image.open(image)
    img_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    image = img_transform(img_pl)
    return image
                      
                      
def predict(image_path, model, topk=5):
    
   
    
    
    model.to('cuda')
    model.eval()
    img = process_image(image_path).numpy()
    img = torch.from_numpy(np.array([img])).float()
    
    with torch.no_grad():
        logps = model.forward(img.cuda())
        
    probability = torch.exp(logps).data
    
    return probability.topk(topk)
                      