Project 3 in the self-driving nanodegree at Udacity consist in training a model to mimic driver behavior.
The goal is to first drive around the track yourself to generate the training set, and later train a network that is capable of completing the same track on autonomous mode.

### Creating and Preprocessing of the Training Set
To capture good driving behavior, I first drove two laps around the track. 
While I drive, the simulator records the steering angle with three images, one from a center camera, right camera and left camera. 
The location of these images with its steering angle gets at the end of the recoring saved to a csv, making it easy for me later to load the images to train the network.


### Model Architecture and Training Strategy

My idea was to build an CNN similar to the one used by NVIDIA in their Self-Driving Car. 
Since the network had proven earlier by others to work on self-driving cars, I considered it to be a safe choice to choose a similar approach.
The network starts with an input and normalization layer with images having the shape (200x66x3), before passing it through 5 convolutional layers and at the end three fully connected layers.

Before starting training my model, I split the training set into a training and validation set, where the validation set would be used by TensorFlow to validate how my model was doing during training.
During training, I used an generator to load the images by chunks as there are too many images to be loaded all at once. I used a batch size of 32, meaning the generator gathered a group of 32 images and labels together and returned them group wise to the training method.
This way, the generator could keep loading images at the same time as the model was being trained. For each line in the data, the center camera, right and left camera had each their three images associated with one steering angle. 
Therefore I randomly choose one image of the camera at each iteration, and randomly another one at the next. I cropped the images, to view only the regions of interest before resizing them to the appropriate size input requested by the model (200x66x3).
Also, since the left and right images are placed on the side of the car, I added an offset to the steering angle to make all the images comparable. I first tested with an offset of 0.200, but saw I needed to increase it, and increased it to 0.275.

I first trained the model a few iterations and noticed that the mean squared error of the validation set stopped dropping, which could indicate that the model was overfitting on the training set.
To handle this, I added a dropout layer after the first fully-connected layer with a 50% dropout percentage. 

After adding the extra dropout layer, the model instantly improved and the validation set error dropped further. 

#### Final Model Architecture



| Layer # | Type | Input | Output | Activation
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
1 | Input | 200x66x3 | 200x66x3 | -
1 | Normalization | 200x66x3 | 200x66x3 | -
2 | Convolutional | 200x66x3 | 98x31x24 | relu
3 | Convolutional | 98x31x24 | 47x14x36 | relu
4 | Convolutional | 47x14x36 | 22x5x48 | relu
5 | Convolutional | 22x5x48 | 20x3x64 | relu
5 | Convolutional | 20x3x64 | 18x1x64 | relu
6 | Flatten | 18x1x64 | 100 | -
7 | Fully Connected | 100 | 50 | relu
8 | 50% Dropout | - | - | - |
9 | Fully Connected | 50 | 10 | relu
10 | Fully Connected | 10 | 1 | -



### Final Results

With this final network, I trained the model with the images from the recorded images and drove successfully the car around the entire track.
I wanted to test how it ran when increasing the speed, and tried setting it to 20 mph, to see if it would still drive successfully. 
With 20 mph it failed to complete the first turn, which might be of the simple fact that the simulator is very slow on my computer and the car might be moving too fast for the network to react.
I then sat the speed down to lower speed again and recorded the car driving successful around the track.
