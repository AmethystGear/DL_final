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
        self.conv1 = nn.Conv2d(4, 8, 3, stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.batch3 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 3, 3, stride=1, padding=1)
        self.accuracy = None

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batch2(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batch3(x)
        x = self.conv3(x)
        return x
        
    def loss(self, pred, original_image, x_start, x_end, y_start, y_end):
        total_loss = 0.0
        for i in range(len(x_start)):
            total_loss += F.mse_loss(pred[i][0:3, x_start[i]:x_end[i], y_start[i]:y_end[i]], original_image[i][0:3, x_start[i]:x_end[i], y_start[i]:y_end[i]])
        return total_loss
        #return F.mse_loss(pred[0:3, x_start:x_end, y_start:y_end], original_image[0:3, x_start:x_end, y_start:y_end])


def construct_output(filename, pred, target, x_start, x_end, y_start, y_end):
    pred_img = transforms.ToPILImage()(pred)
    target_img = transforms.ToPILImage()(target)
    mixed = target_img.copy()
    mixed_pix = mixed.load()
    pred_pix = pred_img.load()
    for i in range(x_start, x_end):
        for j in range(y_start, y_end):
            mixed_pix[i, j] = pred_pix[i, j]

    for i in range(pred_img.size[0]):
        for j in range(pred_img.size[1]):
            if i < x_start or i >= x_end or j < y_start or j>= y_end:
                pred_pix[i, j] = (0, 0, 0)

    concat = Image.new('RGB', (pred_img.width * 3, pred_img.height))
    concat.paste(pred_img, (0, 0))
    concat.paste(target_img, (pred_img.width, 0))
    concat.paste(mixed, (pred_img.width * 2, 0))
    concat.save(os.path.join('model_output', filename))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device', device)

train_data, test_data = [RedactoDataset(os.path.join('dataset', x)) for x in ['train', 'test']]
train_loader, test_loader = [torch.utils.data.DataLoader(x, batch_size = BATCH_SIZE, shuffle=True) for x in [test_data, train_data]]

model = RedactoNet()

losses = []
epoch = 0
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
while epoch < 16:
    train_loss_total = 0

    index = 0
    os.mkdir('model_output')
    random_samples = [(random.randint(0, len(train_loader) - 1), random.randint(0, BATCH_SIZE - 1)) for _ in range(4)]
    for (observation, target, x_start, x_end, y_start, y_end) in train_loader:
        model.zero_grad()
        pred = model.forward(observation)

        train_loss = model.loss(pred, target, x_start, x_end, y_start, y_end)

        for random_sample in random_samples:
            if random_sample[0] == index:
                batch_index = random_sample[1]
                image_name = "{}_{}.png".format(epoch, index)
                construct_output(
                    image_name, 
                    pred[batch_index], 
                    target[batch_index], 
                    x_start[batch_index], 
                    x_end[batch_index], 
                    y_start[batch_index], 
                    y_end[batch_index]
                )

        train_loss.backward()
        optimizer.step()
        train_loss_total += train_loss
        index += 1

    losses.append((train_loss_total/len(train_loader)).item())
    print("epoch = " + str(epoch) + ", loss = " + str(train_loss_total))
    epoch += 1

