import numpy
import os
import random
from threading import Thread

from PIL import Image
from PIL import ImageFilter

from scipy import misc

from sklearn.feature_extraction import image as fe



BATCH_WIDTH = BATCH_HEIGHT = 4
ALEXNET_WIDTH = ALEXNET_HEIGHT = 227

NUM_TRIALS = 1000

VESSEL_CLASS = 1
NON_VESSEL_CLASS = 0

TRAINING_PATH = "./DRIVE/training/images/"
LABEL_PATH    = "./DRIVE/training/1st_manual/"

STORE_FEATURE_PATH = 'dataset/features.npy'
STORE_LABEL_PATH = 'dataset/labels.npy'

STORE_TEST_PATH = 'dataset/test.npy'
STORE_TEST_PATH_LABEL = 'dataset/test_labels.npy'
resize = True

available_threads = 8;

class Drive:
    def __init__(self,train):
        self.train = train

class Dataset:
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        self.size = len(inputs)
        self.current_batch = 0
    
    def next_batch(self):
        batch = self.inputs[self.current_batch], self.labels[self.current_batch]
        self.current_batch = (self.current_batch + 1) % len(self.inputs)
        return batch


#counts the number of black pixel in the batch
def mostlyBlack(image):
    pixels = numpy.array(image.getdata())
    size = image.size[0]*image.size[1]
    nblack = 0
    for pixel in pixels:
      if pixel < 10: 
	nblack+=1
      
    return float(nblack)/size > 0.98

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
    image = image.crop((x, y, x + BATCH_WIDTH, y + BATCH_HEIGHT)).split()[1]
    label = label.crop((x, y, x + BATCH_WIDTH, y + BATCH_HEIGHT))#.split()[0]
    
    return image, label

def shuffle(a, b):
    combined = zip(a, b)
    random.shuffle(combined)

    a[:], b[:] = zip(*combined)

#creates NUM_TRIALS images from a dataset
def fill(images_path, label_path, files, label_files, images, labels, label_class, num, thread):
    print "devo farne {}".format(num)
    t = 0
    while t < num:
        index = random.randrange(0, len(files))
        if files[index].endswith(".tif"):
            image_filename = images_path + files[index]
            label_filename = label_path + label_files[index]
            image = Image.open(image_filename)
            label = Image.open(label_filename)
            
            #image = image.filter(ImageFilter.EDGE_ENHANCE)
	    
            image, label = cropImage(image, label)
            
            if not mostlyBlack(image):
	      if label_class == VESSEL_CLASS and isVessel(label):
		labels.append([label_class])
		if resize: image = misc.imresize(image, (ALEXNET_WIDTH, ALEXNET_HEIGHT))
		images.append(to_rgb(numpy.array(image), image.size))
		t = t + 1
		#image.save("images/{}_{}.jpg".format(t,thread))
		#label.save("images/{}_{}_.jpg".format(t,thread))
		print "{} of {}".format(t,num)

	      if label_class == NON_VESSEL_CLASS and not isVessel(label) and mostlyBlack(label):
		labels.append([label_class])
		if resize: image = misc.imresize(image, (ALEXNET_WIDTH, ALEXNET_HEIGHT))
		images.append(to_rgb(numpy.array(image), image.size))
		#image.save("images/{}_{}.jpg".format(t,thread))
		#label.save("images/{}_{}_.jpg".format(t,thread))
		t = t + 1
		print "{} of {}".format(t,num)

 
def create_dataset():
    print "creating dataset..."
    files = os.listdir(TRAINING_PATH)
    label_files = os.listdir(LABEL_PATH)
    
    print len(files)
    
    images = []
    labels = []

    
    print "vessels"
    
    list_threads = []

    num = NUM_TRIALS*35/100
    
    white = num
    for i in range(0,available_threads - 1):
      num = num - white/available_threads
      list_threads.append(Thread(target=fill, args=(TRAINING_PATH, LABEL_PATH, files, label_files, images, labels, VESSEL_CLASS, white/available_threads,i)))
    list_threads.append(Thread(target=fill, args=(TRAINING_PATH, LABEL_PATH, files, label_files, images, labels, VESSEL_CLASS, num,8)))

    [t.start() for t in list_threads]
    [t.join() for t in list_threads]
    
    print "NON vessels"
    fill(TRAINING_PATH, LABEL_PATH, files, label_files, images, labels, NON_VESSEL_CLASS, NUM_TRIALS - white,9)
    
    shuffle(images, labels)

    train = Dataset(images, labels)
    print "dataset created"
    
    return Drive(train)
    
test_batch_size = 10;
def prepare_image(image_filename, label_filename):
    print "preparing image"
    
    images = []
    labels = []
    image = Image.open(image_filename)
    label = Image.open(label_filename) 
    
    #to remove
    #box = (214, 131, 214 + test_batch_size, 131 + test_batch_size)
    #image = image_.crop(box)
    #label = label_.crop(box)
    ##
  
    imgwidth, imgheight = image.size
    for i in range(0,imgheight):
        for j in range(0,imgwidth):
            box = (j, i, j + BATCH_WIDTH, i + BATCH_HEIGHT)
            im = image.crop(box).split()[1]  
            if resize: im = misc.imresize(im, (ALEXNET_WIDTH, ALEXNET_HEIGHT))
            images.append(to_rgb(numpy.array(im), [100,100]))
            labels.append(VESSEL_CLASS) if isVessel(label.crop(box)) else labels.append(NON_VESSEL_CLASS)
            print len(images)
            
    test = Dataset(images, labels)
    return Drive(test)

def save_as_image(pixels,labels, size):
    
    #pixels = [x * 255 for x in pixels]

    #im = fe.reconstruct_from_patches_2d(to_rgb1a(pixels),(test_batch_size,test_batch_size))
    print numpy.shape(pixels)
    im = Image.fromarray(to_rgb1a(pixels, labels))
    im.show('test.png')
    
def to_rgb1a(pixels, labels):
    import math
    size = math.sqrt(len(pixels))
    pic = numpy.reshape(pixels, (584, 565))
    w, h = numpy.shape(pic)
    res = numpy.empty((w, h, 3), dtype=numpy.uint8)
    

    k = 0
    for i in range(0, w):
      for j in range(0, h):
	if(pixels[k] == 1 and labels[k] == 1):
	  res[i][j][0] = res[i][j][1] = res[i][j][2] = 255
	if(pixels[k] == 0 and labels[k] == 0):
	  res[i][j][0] = res[i][j][1] = res[i][j][2] = 0
	if(pixels[k] == 1 and labels[k] == 0):
	  res[i][j][0] = 255
	  res[i][j][1] = res[i][j][2] = 0
	if(pixels[k] == 0 and labels[k] == 1):
	  res[i][j][1] = 255
	  res[i][j][0] = res[i][j][2] = 0
	res[i][j][0] = res[i][j][1] = res[i][j][2] = pixels[k]*255
	k = k+1
    
    return res
  
def to_rgb(pixels, size):

  w, h = numpy.shape(pixels)
  res = numpy.empty((w, h, 3), dtype=numpy.uint8)
    
  res[:,:,2] = res[:,:,1] = res[:,:,0] = pixels
    
  return res
