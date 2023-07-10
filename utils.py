import torch
import torchvision
from PIL import Image
import pandas as pd


def predict(image_path: str, model) -> tuple:
    """
    Make predictions for input image

    :param image_path: path to image (str)
    :param model: pretrained model for make predictions (torch model)
    :return: tuple consists of list with probabilities and list with predicted classes
    """
    classes = ['apple', 'banana', 'bean', 'beetroot', 'bell pepper', 'bitter_gourd', 'bottle_gourd',
               'brinjal', 'broccoli', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper',
               'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon',
               'lettuce', 'mango', 'onion', 'orange', 'papaya', 'paprika', 'pear', 'peas', 'pineapple',
               'pomegranate', 'potato', 'pumpkin', 'raddish', 'radish', 'soy beans', 'spinach', 'sweetcorn',
               'sweetpotato', 'tomato', 'turnip', 'watermelon']

    input_image = Image.open(image_path) # get user image

    data_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=256,
                                      interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        torchvision.transforms.RandomCrop(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ]) # set image augmentations
    input_image = data_transforms(input_image) # apply image augmentations

    model.eval() # set model on eval mode
    with torch.no_grad():
        input = torch.unsqueeze(input=input_image, dim=0) # add dimension with batch size  
        output = torch.softmax(model(input), dim=1) # get predictions  

    df_pred = pd.DataFrame({
        'classes': classes,
        'preds': torch.squeeze(input=output, dim=0).numpy()
    }).sort_values('preds', ascending=False).head(5) # create df with classes and preds

    prob_list = [round(pred * 100, 2) for pred in df_pred['preds'].tolist()] # create list with probs
    classes_list = df_pred['classes'].tolist() # create list with classes 
    return prob_list, classes_list


def allowed_file(filename: str) -> bool:
    """
    Check extension of input file
    :param filename: name of file (str)
    :return: result of checking (True or False)
    """
    result_extension = filename.split('.')[-1] in ('jpg', 'jpeg', 'png', 'jfif') # check extension 
    return result_extension


