import numpy
import os
import random

from PIL import Image
from scipy import misc

BATCH_WIDTH = BATCH_HEIGHT = 28
ALEXNET_WIDTH = ALEXNET_HEIGHT = 227

NUM_TRIALS = 1000

VESSEL_CLASS = 1
NON_VESSEL_CLASS = 0

TRAINING_PATH = "./DRIVE/training/images/"
LABEL_PATH    = "./DRIVE/training/1st_manual/"

STORE_FEATURE_PATH = 'dataset/features.npy'
STORE_LABEL_PATH = 'dataset/labels.npy'

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

def shuffle(a, b):
    combined = zip(a, b)
    random.shuffle(combined)

    a[:], b[:] = zip(*combined)
    
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
            
            if not mostlyBlack(image):
                if label_class == VESSEL_CLASS and isVessel(label):
                    labels.append([label_class])
                    image = misc.imresize(image, (ALEXNET_WIDTH, ALEXNET_HEIGHT))
                    images.append(numpy.array(image))
                    t += 1
                if label_class == NON_VESSEL_CLASS and not isVessel(label):
                    labels.append([label_class])
                    image = misc.imresize(image, (ALEXNET_WIDTH, ALEXNET_HEIGHT))
                    images.append(numpy.array(image))
                    t += 1
                    
def create_dataset():
    print "creating dataset..."
    files = os.listdir(TRAINING_PATH)
    label_files = os.listdir(LABEL_PATH)
    
    images = []
    labels = []
    print "vessels"
    fill(TRAINING_PATH, LABEL_PATH, files, label_files, images, labels, VESSEL_CLASS, NUM_TRIALS/2)
    print "NON vessels"
    fill(TRAINING_PATH, LABEL_PATH, files, label_files, images, labels, NON_VESSEL_CLASS, NUM_TRIALS/2)
    
    shuffle(images, labels)

    train = Dataset(images, labels)
    print "dataset created"
    
    return Drive(train)
    

def prepare_image(image_filename, label_filename):

    images = []
    labels = []
    image = Image.open(image_filename)
    label = Image.open(label_filename) 
        
    imgwidth, imgheight = image.size
    for i in range(0,imgheight, BATCH_HEIGHT):
        for j in range(0,imgwidth, BATCH_WIDTH):
            box = (j, i, j + BATCH_WIDTH, i + BATCH_HEIGHT)
            images.append(image.crop(box))
            labels.append(VESSEL_CLASS) if isVessel(label.crop(box)) else labels.append(NON_VESSEL_CLASS)
            
    test = Dataset(images, labels)
    return Drive(test)

def save_as_image(images, size):
    rescaled = numpy.reshape(images, size)
    im = Image.fromarray(rescaled)
    im.show()