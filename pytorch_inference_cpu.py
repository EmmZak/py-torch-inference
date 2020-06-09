from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import time
from torchvision import models
import copy
import cv2
from time import time, sleep
from glob import glob
import numpy as np
from threading import Thread
from PIL import Image

gpu = torch.cuda.is_available()
device = torch.device("cuda:0")
print(device)
print(gpu)

model_name = "general"
dataset_path = model_name + "/"


classes = ["bottleopaque", "bottletrans", "box", "boxopaque", "boxtrans", "canette", "chips", "conserve", "filme", "hand", "nothing", "petecrase", "pp"]
#classes = ["cap", "nocap", "nothing"]

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

model = models.mobilenet_v2(pretrained=True)
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(
        in_features=1280,
        out_features=len(classes)
    ),
    torch.nn.Sigmoid()
)
model.load_state_dict(torch.load(
    "mobilenet_models/best_weights_" + model_name + ".pth"))

model = model.to(device)

model.eval()

cam = cv2.VideoCapture(0)

while True:

    ok, im = cam.read()
    im = im[75:425, 50:640]

    cv2.imshow("test", im)
    cv2.waitKey(1)

    im = transform(im).unsqueeze(0)

    im = im.to(device)

    output = model(im)

    _, preds = torch.max(output, 1)

    c = classes[preds]

    print(c)
