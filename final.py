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
        image = Image.open(filename)
        image = transforms.ToTensor()(image)
        
        size = image.size()
        redacto = torch.cat((image, torch.zeros(1, size[1], size[2])), 0)

        for c in range(0, 4):
            for i in range(0, REDACT_SIZE):
                for j in range(0, REDACT_SIZE):
                    if c < 3:
                        redacto[c, x + i, y + j] = 0.0
                    else:
                        redacto[c, x + i, y + j] = 1.0

        return redacto, image, x, x + REDACT_SIZE, y, y + REDACT_SIZE
        

class RedactoNet(nn.Module):
    def __init__(self):
        super(RedactoNet, self).__init__()
        #self.batch1 = nn.BatchNorm2d(4)
        self.conv1 = nn.Conv2d(4, 16, 3, stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 7, stride=1, padding=3)
        self.batch3 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 15, stride=1, padding=7)
        self.batch4 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, 7, stride=1, padding=3)
        self.batch5 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 3, 3, stride=1, padding=1)
        self.accuracy = None

    def forward(self, x):
        #x = self.batch1(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batch2(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batch3(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batch4(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batch5(x)
        x = self.conv5(x)
        return x
        
    def loss(self, pred, original_image, x_start, x_end, y_start, y_end):
        total_loss = 0.0
        for i in range(len(x_start)):
            total_loss += F.mse_loss(pred[i][0:3, x_start[i]:x_end[i], y_start[i]:y_end[i]], original_image[i][0:3, x_start[i]:x_end[i], y_start[i]:y_end[i]])
        return total_loss
        #return F.mse_loss(pred[0:3, x_start:x_end, y_start:y_end], original_image[0:3, x_start:x_end, y_start:y_end])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device ', device)

train_data, test_data = [RedactoDataset(os.path.join('dataset', x)) for x in ['train', 'test']]
train_loader, test_loader = [torch.utils.data.DataLoader(x, batch_size = BATCH_SIZE, shuffle=True) for x in [test_data, train_data]]

model = RedactoNet()

losses = []
epoch = 0
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
while epoch < 16:
    train_loss_total = 0
    for (observation, target, x_start, x_end, y_start, y_end) in train_loader:
        model.zero_grad()
        train_loss = model.loss(model.forward(observation), target, x_start, x_end, y_start, y_end)
        train_loss.backward()
        optimizer.step()
        train_loss_total += train_loss
    losses.append((train_loss_total/len(train_loader)).item())
    print("epoch = " + str(epoch) + ", loss = " + str(train_loss_total))
    epoch += 1

