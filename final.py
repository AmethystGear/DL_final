import os
import torch
from torch import nn
import torch.nn.functional as F
import random
from PIL import Image
from torchvision import transforms
import shutil
from pathlib import Path
import sys
torch.set_printoptions(threshold=10_000, linewidth=200)

# location where we keep the redacted images (train, test)
DATASET_LOC = 'dataset'

# number of redacted images we make from one image
IMG_SAMPLES = 4

# we remove a REDACT_SIZE * REDACT_SIZE square of pixels from the image
REDACT_SIZE = 2

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
                        redacto[c, y + i, x + j] = 0.0
                    else:
                        redacto[c, y + i, x + j] = 1.0

        return Path(filename).name, redacto, image, x, y
        

class RedactoNet(nn.Module):
    def __init__(self):
        super(RedactoNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 3, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        return x
        
    def loss(self, pred, target, x, y): 
        def redact_part(im):
            return im[0][0:3, x[0]: x[0] + REDACT_SIZE, y[0]:y[0] + REDACT_SIZE]

        return F.mse_loss(redact_part(pred), redact_part(target))

def concat_images_horizontally(images):
    w = sum([image.height for image in images])
    h = images[0].height
    concat = Image.new('RGB', (w, h))

    curr_w = 0
    for i in range(len(images)):
        concat.paste(images[i], (curr_w, 0))
        curr_w += images[i].width

    return concat


def construct_output(filename, pred, target, observation, x, y):
    observation_img = transforms.ToPILImage()(observation)
    pred_img = transforms.ToPILImage()(pred)
    target_img = transforms.ToPILImage()(target)
    pred_all = pred_img.copy()
    mixed = target_img.copy()
    mixed_pix = mixed.load()
    pred_pix = pred_img.load()
    for i in range(x, x + REDACT_SIZE):
        for j in range(y, y + REDACT_SIZE):
            mixed_pix[i, j] = pred_pix[i, j]

    for i in range(pred_img.size[0]):
        for j in range(pred_img.size[1]):
            if i < x or i >= x + REDACT_SIZE or j < y or j>= y + REDACT_SIZE:
                pred_pix[i, j] = (0, 0, 0)

    target_isolate = target_img.copy()
    target_isolate_pix = target_isolate.load()
    for i in range(pred_img.size[0]):
        for j in range(pred_img.size[1]):
            if i < x or i >= x + REDACT_SIZE or j < y or j>= y + REDACT_SIZE:
                target_isolate_pix[i, j] = (0, 0, 0)

    concat = concat_images_horizontally([observation_img, pred_img, target_isolate, pred_all, target_img, mixed])
    concat.save(os.path.join('model_output', filename))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device', device)

train_data, test_data = [RedactoDataset(os.path.join('dataset', x)) for x in ['train', 'test']]
train_loader, test_loader = [torch.utils.data.DataLoader(x, shuffle=True) for x in [train_data, test_data]]

model = RedactoNet()

shutil.rmtree('model_output')
os.mkdir('model_output')

losses = []
epoch = 0
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
while epoch < 16:
    train_loss_total = 0

    index = 0
    percent = 0
    random_samples = [random.randint(0, len(train_loader) - 1) for _ in range(4)]
    for (name, observation, target, x, y) in train_loader:
        model.zero_grad()
        pred = model.forward(observation)

        train_loss = model.loss(pred, target, x, y)

        if index in random_samples:
            image_name = "{}_{}_{}".format(epoch, index, name[0])
            construct_output(image_name, pred[0], target[0], observation[0], x[0], y[0])

        train_loss.backward()
        optimizer.step()
        train_loss_total += train_loss
        index += 1
        if int(index / len(train_loader) * 100) > percent:
            percent += 1
            print('#', end='')
            sys.stdout.flush()




    print('')


    losses.append((train_loss_total/len(train_loader)).item())
    print("epoch = " + str(epoch) + ", loss = " + str(train_loss_total))
    epoch += 1

