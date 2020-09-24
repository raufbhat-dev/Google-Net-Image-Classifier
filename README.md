# intel_image_classification

**Image classification by implementing googLeNet**

This source code mimics the googlenet that won the ILSVRC14 challenge with slight modifications in the fully connected layers in all the 3 outputs to accomodate the use case described below.

**Hardware Peak Utilisation**
CPU: 32 core
Virtual Memory: 25 GB
RES Memory: 15 GB
Runtime: ~35 mins

**software**
python: 3.5.2
tensorflow:2.3.0

**Data Set**
source: Kaggle 
userid: puneet6060 
data-set: intel-image-classification

**Normenclature**:
bell1: The bottom most fully connected FC NN emerging from the inception module 4a as described in the googlnet incarnation table.
bell2: The middle fully connecneted FC NN emerging from the output of inception module 4d as described in the googlnet incarnation table.
mainbell: The top most fully connecneted FC NN emerging from the last inception module 5b as described in the googlnet incarnation table.

**Optimiser**: Adam
leraning rate: 0.001 decaying by 50% after each epoch
beta1(first moment): 0.09
beta2 (second moment):0.999
loss function: SparseCategoricalCrossentropy with logits
epoch: 3

**Note**:
All the FC NN have been modified to accomodate this use case.
The Conv and Pooling layers remain as described in the googlenet paper.


**Observations**:
epoch 1: Accuracy for both training and validation is in the sequence  bell1 > bell2 > mainbell
epoch 2: Accuracy for both training and validation is in the sequence  bell2 >~ bell1 >~ mainbell
epoch 3: Accuracy for both training and validation is in the sequence  bell2 =~ bell1 =~ mainbell

"~" represents: with a very tiny margin.
