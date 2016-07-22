import numpy
import os
import random

from PIL import Image
from scipy import misc


BATCH_WIDTH = BATCH_HEIGHT = 28
ALEXNET_WIDTH = ALEXNET_HEIGHT = 227

NUM_TRIALS = 10

class Drive:
    def __init__(self,train):
        self.train = train

class Dataset:
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        self.current_batch = 0
    
    def next_batch(self):
        batch = self.inputs[self.current_batch], self.labels[self.current_batch]
        self.current_batch = (self.current_batch + 1) % len(self.inputs)
        return batch


#counts the number of black pixel in the batch
def mostlyBlack(image):
    pixels = image.getdata()
    black_thresh = 50
    nblack = 0
    for pixel in pixels:
        if pixel < black_thresh:
            nblack += 1

    return nblack / float(len(pixels)) > 0.5

#counts the number of white pixel in the batch
def isVessel(label):
    pixels = label.getdata()
    
    white_thresh = 250
    x = BATCH_HEIGHT/2
    y = BATCH_WIDTH/2
    
    pos = (BATCH_HEIGHT*x)+y
    #could be useful compute a mean between pos-1,pos,pos+1 if BATCH_[HEIGHT|WIDTH] % 2 == 0
    pixel = pixels[pos]

    return pixel >= white_thresh
    
#crop the image starting from a random point
def cropImage(image, label):
    width  = image.size[0]
    height = image.size[1]
    x = random.randrange(0, width - BATCH_WIDTH)
    y = random.randrange(0, height - BATCH_HEIGHT)
    image = image.crop((x, y, x + BATCH_WIDTH, y + BATCH_HEIGHT))#.split()[1]
    label = label.crop((x, y, x + BATCH_WIDTH, y + BATCH_HEIGHT))#.split()[0]
    
    return image, label


#creates NUM_TRIALS images from a dataset
def fill(images_path, label_path, files, label_files, images, labels, label_class, num):
    t = 0
    while t < num:
        index = random.randrange(0, len(files))
        if files[index].endswith(".tif"):
            image_filename = images_path + files[index]
            label_filename = label_path + label_files[index]
            image = Image.open(image_filename)
            label = Image.open(label_filename)
            image, label = cropImage(image, label)
            
            if not mostlyBlack(image) and isVessel(label):    
                labels.append([label_class])
                image = misc.imresize(image, (ALEXNET_WIDTH, ALEXNET_HEIGHT))
                images.append(numpy.array(image))
                t += 1

def create_dataset(images_path, label_path):
    files = os.listdir(images_path)
    label_files = os.listdir(label_path)
    
    images = [];
    labels = [];
    fill(images_path, label_path, files, label_files, images, labels, 0, NUM_TRIALS/2)
    fill(images_path, label_path, files, label_files, images, labels, 1, NUM_TRIALS/2)
    
    shuffle(images, labels)

    train = Dataset(images, labels)
    return Drive(train)


def shuffle(a, b):
    combined = zip(a, b)
    random.shuffle(combined)

    a[:], b[:] = zip(*combined)