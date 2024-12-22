from  main import  imshow
# Numpy
import numpy as np
from torch.utils.data import DataLoader
# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from  models.vgg import  VGGAutoEncoder, get_configs
import models.builer as builder

# Torchvision
import torchvision
import torchvision.transforms as transforms
import argparse
import utils
from  attention import congnitive_distillation
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Trainer for auto encoder')
    parser.add_argument('--arch', default='vgg16', type=str,
                        help='backbone architechture')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--parallel', type=int, default=1,
                        help='1 for parallel, 0 for non-parallel')
    parser.add_argument("--valid", action="store_true", default=True,
                        help="Perform validation only.")
    parser.add_argument('--resume', type=str, default=' ')
    parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU id to use')
    parser.add_argument('--target', type=int, default=100)

    args = parser.parse_args()
    args.parallel = 0
    args.batch_size = 1
    args.workers = 0
    return args

args = get_args()
utils.init_seeds(1, cuda_deterministic=False)
# Load data
transform = transforms.Compose([
       transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
   ])
normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

trainset = torchvision.datasets.ImageNet(root=' ', split='test', transform=transform)
trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)

autoencoder = builder.BuildAutoEncoder(args)
autoencoder = autoencoder.to(device)
utils.load_dict(args.resume, autoencoder)

criterion = F.cross_entropy

model1 = torchvision.models.resnet50(pretrained=True)
model1.eval()
model1.to(device)

model2 = torchvision.models.vgg16(pretrained=True)
model2.eval()
model2.to(device)

model3 = torchvision.models.densenet121(pretrained=True)
model3.eval()
model3.to(device)

features = []
epsilon = 0.2


for images, labels in trainloader:

    if labels.item() != args.target:
        continue

    images, labels = images.to(device), labels.to(device)


    A = []
    inputs = normalize(images)
    #
    outputs = model1(images)
    outputs = F.softmax(outputs, dim=1)
    A.append(outputs)

    outputs = model2(images)
    outputs = F.softmax(outputs, dim=1)
    A.append(outputs)

    outputs = model3(images)
    outputs = F.softmax(outputs, dim=1)
    A.append(outputs)

    outputs = sum(A) / len(A)

    if outputs[0][args.target] >= 0.7 :
        imshow(torchvision.utils.make_grid(images.data))
        with torch.no_grad() :
            feature = autoencoder.module.get_feature(images)
            features.append(feature)
    if len(features) >= 10 :
          break


feature = sum(features) / len(features)
feature.requires_grad_()


optimizer = optim.Adam([feature], lr=0.005)


for images, labels in trainloader:

    if labels.item() != args.target:
        continue
    images, labels = images.to(device),labels.to(device)

    for step in range(1500) :
           optimizer.zero_grad()
           decoded = autoencoder.module.decode(feature)

           perturb = torch.rand(images.shape).to(device)
           decoded = decoded +  epsilon * perturb
           decoded = normalize(decoded)


           outputs1 = model1(decoded)
           outputs2 = model2(decoded)
           outputs3 = model3(decoded)
           loss = criterion(outputs1,labels)  + criterion(outputs2, labels) + criterion(outputs3, labels)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           if (step + 1) % 100 == 0:
               print('Feature Loss: ', loss.item())

    decoded_img = autoencoder.module.decode(feature)

    decoded_img_n = normalize(decoded_img)
    outputs = model1(decoded_img_n)
    _, predicted = outputs.max(1)
    print('predict: ',predicted)
    print('label:   ', labels)

    outputs = model2(decoded_img_n)
    _, predicted = outputs.max(1)
    print('predict: ',predicted)
    print('label:   ', labels)

    outputs = model3(decoded_img_n)
    _, predicted = outputs.max(1)
    print('predict: ',predicted)
    print('label:   ', labels)


    np.savez(' ', array=feature.detach().cpu().numpy())
    exit()








