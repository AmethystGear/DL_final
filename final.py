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
IMG_SAMPLES = 10

# we remove a REDACT_SIZE * REDACT_SIZE square of pixels from the image
REDACT_SIZE = 4

# max random seed we use
RAND_MAX_SEED = 1000000

# number of samples in an epoch
EPOCH_SIZE = 3000

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
            f.write("{}\n".format(random.randint(0, RAND_MAX_SEED)))
            for filename in os.scandir(os.path.join('cifar', folder)):
                filepath = os.path.join('cifar', folder, filename.name)
                for i in range(IMG_SAMPLES):
                    x = 14
                    y = 14
                    f.write("{} {} {}\n".format(filepath, x, y))


# this gets the image and the position to redact, and then redacts 
# in memory. Much nicer than storing all the redacted files on disk
class RedactoDataset(torch.utils.data.Dataset):
    def __init__(self, data_file):
        self.data_file = open(data_file, 'r').readlines()
        seed = int(self.data_file[0])
        torch.manual_seed(seed)
        self.data_file = self.data_file[1:]

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        filename, x, y = self.data_file[idx].split(' ')
        x = int(x)
        y = int(y)
        image = transforms.ToTensor()(Image.open(filename))
        
        size = image.size()
        redacto = torch.cat((image, torch.zeros(1, size[1], size[2])), 0)
        for c in range(0, 4):
            for i in range(0, REDACT_SIZE):
                for j in range(0, REDACT_SIZE):
                    if c < 3:
                        redacto[c, y + i, x + j] = 0.5
                    else:
                        redacto[c, y + i, x + j] = 1.0

        return Path(filename).name, redacto, image, x, y
        
        
class RedactoModel(nn.Module):
    def create_net(self, channels):
        net = nn.Sequential()
        for i in range(0, len(channels) - 2):
            net.append(RedactoLayer((channels[i], channels[i + 1])))

        return net

    def __init__(self, channels):
        super(RedactoModel, self).__init__()
        self.net = self.create_net(channels)
        self.last_conv_layer = nn.Conv2d(channels[len(channels) - 2], channels[len(channels) - 1], 3, stride=1, padding=1)

    def forward(self, x):
        for layer in self.net.children():
            redacto = x[:, -1].unsqueeze(1)
            x = layer(x)
            x = torch.cat((x, redacto), 1)

        return self.last_conv_layer(x)

    def loss(self, pred, target): 
        return F.mse_loss(pred, target)


class RedactoLayer(nn.Module):
    def create_net(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels[0], channels[1] - 1, 3, stride=1, padding=1),
            nn.ReLU()
        )

    def __init__(self, channels):
        super(RedactoLayer, self).__init__()
        self.main_net = self.create_net(channels)

    def forward(self, x):
        return self.main_net(x)
        

def concat_images_horizontally(images):
    w = images[0].width
    h = images[0].height
    concat = Image.new('RGB', (w * len(images), h))

    for i in range(len(images)):
        if images[i] != None:
            concat.paste(images[i], (i * w, 0))

    return concat

def construct_output(folder, filename, pred, target, target_redact, observation, x, y):
    observation_img = transforms.ToPILImage()(observation)
    pred_img = transforms.ToPILImage()(pred)
    target_img = transforms.ToPILImage()(target)
    target_redact_img = transforms.ToPILImage()(target_redact)

    mixed = target_img.copy()
    mixed_pix = mixed.load()
    pred_pix = pred_img.load()
    for i in range(REDACT_SIZE):
        for j in range(REDACT_SIZE):
            mixed_pix[x + i, y + j] = pred_pix[i, j]

    concat = concat_images_horizontally([observation_img, None, None, target_img, mixed])

    concat.paste(pred_img, (observation_img.width + x, y))
    concat.paste(target_redact_img, (observation_img.width * 2 + x, y))
    concat.save(os.path.join(folder, filename))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device', device)

train_data, test_data = [RedactoDataset(os.path.join('dataset', x)) for x in ['train', 'test']]
train_loader, test_loader = [torch.utils.data.DataLoader(x, shuffle=True) for x in [train_data, test_data]]

model = RedactoModel([4, 8, 6, 3])

if os.path.exists('model_output'):
    shutil.rmtree('model_output')

os.mkdir('model_output')

losses = []
epoch = 0
optimizer = torch.optim.ASGD(model.parameters(), lr=0.05)
while epoch < 50:
    train_loss_total = 0

    percent = 0
    random_samples = [random.randint(0, EPOCH_SIZE - 1) for _ in range(4)] + [0]

    i = 0
    train_loader_iter = iter(train_loader)
    for e in range(0, EPOCH_SIZE):
        (name, observation, target, x, y) = next(train_loader_iter)
        i += 1
        i %= len(train_loader)
        if i == 0:
            train_loader_iter = iter(train_loader)

        def redact_part(im, x, y, w):
            return im[0][0:3, y: y + w, x: x + w]

        model.zero_grad()
        pred = model.forward(observation)

        pred_xy = pred[0].shape[1] // 2 - REDACT_SIZE // 2
        pred_redact = redact_part(pred, pred_xy, pred_xy, REDACT_SIZE)
        target_redact = redact_part(target, x[0], y[0], REDACT_SIZE)

        train_loss = model.loss(pred_redact, target_redact)
        if e in random_samples:
            image_name = "{}_{}_{}".format(epoch, e, name[0])
            construct_output('model_output', image_name, pred_redact, target[0], target_redact, observation[0], x[0], y[0])

        train_loss.backward()
        optimizer.step()
        train_loss_total += train_loss
        
        if int(e / EPOCH_SIZE * 100) > percent:
            percent += 1
            print('#', end='')
            sys.stdout.flush()


    print('')
    avg_loss = (train_loss_total/EPOCH_SIZE).item()
    losses.append(avg_loss)
    print("epoch = " + str(epoch) + ", loss = " + str(avg_loss))
    epoch += 1


if os.path.exists('model_test_output'):
    shutil.rmtree('model_test_output')

os.mkdir('model_test_output')

ind = 0
loss = 0
random_samples = [random.randint(0, len(test_loader) - 1) for _ in range(20)]
for (name, observation, target, x, y) in test_loader:
    pred = model.forward(observation)
    pred_xy = pred[0].shape[1] // 2 - REDACT_SIZE // 2
    pred_redact = redact_part(pred, pred_xy, pred_xy, REDACT_SIZE)
    target_redact = redact_part(target, x[0], y[0], REDACT_SIZE)
    loss += model.loss(pred_redact, target_redact)

    if ind in random_samples:
        image_name = "{}_{}".format(ind, name[0])
        construct_output('model_test_output', image_name, pred_redact, target[0], target_redact, observation[0], x[0], y[0])

    ind += 1

print('test loss: ', loss / len(test_loader))
