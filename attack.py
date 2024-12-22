from  main import  imshow
# Numpy
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Torchvision
import torchvision
import torchvision.transforms as transforms
from  models.vgg import  VGGAutoEncoder, get_configs
from torch.utils.data import DataLoader
import argparse
import models.builer as builder
import utils
from  attention import spatial_attention_map
import os
import torchmetrics

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def total_variation(tensor):
    _, _, height, width = tensor.size()
    tv_h = torch.sum(torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :]))
    tv_w = torch.sum(torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1]))
    total_variation = tv_h + tv_w

    return total_variation


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
    parser.add_argument('--resume', type=str, default='  ')
    parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU id to use')
    parser.add_argument('--target', type=int, default=100)
    args = parser.parse_args()
    args.parallel = 0
    args.batch_size = 1
    args.workers = 0
    return args

args = get_args()
print(args.gpu)
utils.init_seeds(719, cuda_deterministic=False)

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
autoencoder.eval()
criterion = F.cross_entropy
preception = nn.MSELoss()

ssim_calculator = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)




model1 = torchvision.models.resnet50(pretrained=True)
model1.eval()
model1.to(device)

model2 = torchvision.models.vgg16(pretrained=True)
model2.eval()
model2.to(device)

model3 = torchvision.models.densenet121(pretrained=True)
model3.eval()
model3.to(device)

idx =0
transparency = 1.0
target = args.target

for images, labels in trainloader:
    if labels.item() == target:
        continue

    idx += 1
    images, labels = images.to(device), labels.to(device)

    adv_label = torch.zeros(labels.shape, dtype=torch.long).to(device)
    adv_label.fill_(target)

    data = np.load(' ')
    array = data['array']
    feature = torch.from_numpy(array).to(device)

    with torch.no_grad() :
         org = autoencoder.module.get_feature(images)
         org = org.to(device)

    org, sam = spatial_attention_map(org, labels, autoencoder, [model1, model2, model3], criterion)

    print(sam.shape)

    alpha = torch.rand(org.shape).to(device)
    mask = torch.rand(images.shape).to(device)

    for step in range(1500) :
        alpha.requires_grad_()
        mask.requires_grad_()

        optimizer = optim.Adam( [ {'params': mask, 'lr': 0.002},
        {'params': alpha, 'lr': 0.04}])

        encode = alpha * feature + (1 - sam) * org
        decoded = autoencoder.module.decode(encode)
        decoded = mask * decoded + (1 - mask) * images

        per_loss = torch.norm(mask, p=1)
        tv_loss = total_variation(mask)
        ssim_loss = ssim_calculator(decoded, images)

        decoded = normalize(decoded)

        outputs1, outputs2, outputs3 = model1(decoded), model2(decoded), model3(decoded)

        adv_loss_1 = ( criterion(outputs1, adv_label) + criterion(outputs2, adv_label) + criterion(outputs3, adv_label)) / 3
        adv_loss_2 = (criterion(outputs1, labels) + criterion(outputs2, labels) + criterion(outputs3, labels)) / 3

        adv_loss = 5 * adv_loss_1 -  2 * adv_loss_2

        cog_loss = 0.005 * per_loss +  + 0.002 * tv_loss - 2000 * ssim_loss

        total_loss = adv_loss + cog_loss

        if (step + 1) % 100 ==0:
            print('loss: ', total_loss.item())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        alpha = torch.clamp(alpha.detach(), 0, 1)
        mask = torch.clamp(mask.detach(), 0, 1)

    encode = alpha * feature + sam * org
    decoded = autoencoder.module.decode(encode)
    decoded = transparency *  mask * decoded + (1 - transparency * mask) * images

    inputs = normalize(decoded)

    imshow(torchvision.utils.make_grid(decoded.data))
    imshow(torchvision.utils.make_grid(images.data))
    #

    s = 0

    pil_image = transforms.functional.to_pil_image(decoded.squeeze(0))
    pil_image.save(" " + str(idx) + ".png")





