# transfLearn

Based on: http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

1. Download the weight at http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy (they need to be in the working directory)
2. Download the DRIVE database: http://www.isi.uu.nl/Research/Databases/DRIVE/

## Structure
[myalexnet_forward.py](https://github.com/AldoMarzullo/transfLearn/blob/master/transferLearn/myalexnet_forward.py) contains the pretrained network.

[input_reader.py](https://github.com/AldoMarzullo/transfLearn/blob/master/transferLearn/input_reader.py) is the module to retrieve inputs and labels

[svm.py](https://github.com/AldoMarzullo/transfLearn/blob/master/transferLearn/svm.py) contains the svm classifier
