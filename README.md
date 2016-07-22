# transfLearn

Based on: http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

1. Download the weight at http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy (they need to be in the working directory)
2. Download the DRIVE database: http://www.isi.uu.nl/Research/Databases/DRIVE/

## Structure
[myalexnet_forward.py](https://github.com/AldoMarzullo/transfLearn/blob/master/transferLearn/myalexnet_forward.py) contains the pretrained network.

[input_reader.py](https://github.com/AldoMarzullo/transfLearn/blob/master/transferLearn/input_reader.py) is the module to retrieve inputs and labels

[svm.py](https://github.com/AldoMarzullo/transfLearn/blob/master/transferLearn/svm.py) contains the svm classifier

## Usage
See [svm.py](https://github.com/AldoMarzullo/transfLearn/blob/master/transferLearn/svm.py) as example

Use
`test = input_reader.create_dataset()` to extract `NUM_TRIAL` random batches from the DRIVE dataset of size `BATCH_[HEIGHT|WITDH]` resized to `ALEX_NET_[HEIGHT|WIDTH]`

Use
`test = input_reader.prepare_image(filename,labelname)` to split an image in to n batches of size `BATCH_[HEIGHT|WITDH]` resized to `ALEX_NET_[HEIGHT|WIDTH]`

Use
 `features, labels = net.extract_features(test)` to extract the 'fc7' layer of AlexNet and the corresponding label
 
Use
 `train(features, labels)` to train the svm
 
Use
 `prediction = predict(features)` to classify a new instance
