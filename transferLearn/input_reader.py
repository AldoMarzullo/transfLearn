import numpy
import os
import random

from PIL import Image

BATCH_WIDTH = 227
BATCH_HEIGHT = 227

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
def create_dataset(images_path, label_path):
    files = os.listdir(images_path)
    label_files = os.listdir(label_path)
    
    images = [];
    labels = [];
    t = 0
    while t < NUM_TRIALS:
        index = random.randrange(0, len(files))
        if files[index].endswith(".tif"):
            image_filename = images_path + files[index]
            label_filename = label_path  + label_files[index]
            
            image = Image.open(image_filename)
            label = Image.open(label_filename)
            image, label = cropImage(image, label)
            
            if not mostlyBlack(image):
                images.append(numpy.array(image))
                if t%2 == 0:
                    labels.append([0])
                else:
                    labels.append([1])

                t+=1

    #image = Image.open(images_path + files[1])
    #split_image(image)
    #print numpy.shape(labels[0])
    train = Dataset(images, labels)
    return Drive(train)