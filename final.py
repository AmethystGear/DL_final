import os
import torch
from torch import nn
import torch.nn.functional as F
import random
from PIL import Image
from torchvision import transforms

# location where we keep the redacted images (train, test)
DATASET_LOC = 'dataset'

# number of redacted images we make from one image
IMG_SAMPLES = 4

# we remove a REDACT_SIZE * REDACT_SIZE square of pixels from the image
REDACT_SIZE = 4

BATCH_SIZE = 10

# get cifar if we don't have it
if not os.path.exists('cifar'):
    os.system('wget https://pjreddie.com/media/files/cifar.tar.gz')
    os.system('tar xvzf cifar.tar.gz')
    os.system('rm cifar.tar.gz')

path = os.listdir(os.path.join('cifar', 'test'))[0]
img_size = Image.open(os.path.join('cifar', 'test', path)).size[0]

# generate our train/test dataset by marking out where we're removing chunks
if not os.path.exists(DATASET_LOC):
    os.mkdir(DATASET_LOC)
    folders = ['train', 'test']
    for folder in folders:
        with open(os.path.join(DATASET_LOC, folder), 'w') as f:
            for filename in os.scandir(os.path.join('cifar', folder)):
                filepath = os.path.join('cifar', folder, filename.name)
                for i in range(IMG_SAMPLES):
                    x = random.randint(0, img_size - REDACT_SIZE - 1)
                    y = random.randint(0, img_size - REDACT_SIZE - 1)
                    f.write("{} {} {}\n".format(filepath, x, y))


# this gets the image and the position to redact, and then redacts 
# in memory. Much nicer than storing all the redacted files on disk
class RedactoDataset(torch.utils.data.Dataset):
    def __init__(self, data_file):
        self.data_file = open(data_file, 'r').readlines()

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        filename, x, y = self.data_file[idx].split(' ')
        x = int(x)
        y = int(y)
        img = Image.open(filename)
        pix = img.load()
        for i in range(0, REDACT_SIZE):
            for j in range(0, REDACT_SIZE):
                pix[i + x, j + y] = (0, 0, 0)

        return transforms.ToTensor()(img)

class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.batch1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.batch3 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 3, 3, stride=2, padding=1)
        self.accuracy = None

    def forward(self, x):
        x = self.batch1(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batch2(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batch3(x)
        x = self.conv3(x)
        return x
        
    def loss(self, pred, original_image, x, y):
    

        loss_val = F.cross_entropy(prediction, label.squeeze(), reduction=reduction)
        return loss_val
        



device = torch.device("cuda" if torch.cuda_is_available() else "cpu")
print('Using device ', device)


test_data, train_data = map(['test', 'train'], lambda x: RedactoDataset(os.path.join('dataset', x)))
test_loader, train_loader = map([test_data, train_data], 
    lambda x: torch.utils.data.DataLoader(x, batch_size = BATCH_SIZE, shuffle=True))

model = 


