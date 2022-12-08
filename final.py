import os

REDACT_SIZE = 3

# get cifar if we don't have it
if not os.path.exists('cifar'):
    os.system('wget https://pjreddie.com/media/files/cifar.tar.gz')
    os.system('tar xvzf cifar.tar.gz')


# generate our test dataset by randomly blocking out chunks 