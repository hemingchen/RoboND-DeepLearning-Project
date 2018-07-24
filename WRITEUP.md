[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

# Deep Learning Project: Follow Me

In this project, a deep neural network is trained to identify and track a target in simulation. The model is then used with the simulator to track and follow the target.

[image_0]: ./docs/misc/sim_screenshot.png
![alt text][image_0] 

## 1. Fully Convolutional Network

Compared to a normal CNN that often has a fully connected last layer for classification of the entire image, the last layer of a Fully Convolutional Neural Network (FCN) is another 1x1 convolution layer with a large "receptive field", in order to capture the global context of the scene and tell what are the objects and their approximate locations in the scene. The output will be scene segmentation pixel by pixel rather than object classification of the entire picture.

The structure of FCN is divided into two parts - encoder and decoder. Then encoders extract features from the input image and decoders upscale the output of the encoder to restore original size of the image while classifying each pixel of image.

While a traditional Convolutional Neural Network (CNN) has a fully connected layer at the end, it cannot adapt to inputs with different sizes. On the contrary, a Fully Convolutional Network (FCN) only contains convolutional layers, hence can adapt to various input sizes. This benefit is due to convolution operation only takes a local area of the input at a time, therefore the overall size of the input can be arbitrary and the output is only determined by the shape of the filter. By changing the filter properly, the desired output can be acquired, which makes pixel by pixel semantic segmentation possible.

Due to benefitx of FCN, it is used in this project for object segmentation.


## 2. 1x1 Convolution

The 1x1 convolutions are the foundation of FCN networks. It simply takes an input pixel with all its dimensions and maps it to an output pixel with more or less dimensions. It is often used to reduce the dimension of an input while maintaining its size/resolution, to speed up the processing of the input.


When we convert our last fully connected (FC) layer of the CNN to a 1x1 convolutional layer we choose our new conv layer to be big enough so that it will enable us to have this localization effect scaled up to our original input image size then activate pixels to indicate objects and their approximate locations in the scene as shown in above figure. replacement of fully-connected layers with convolutional layers presents an added advantage that during inference (testing your model), you can feed images of any size into your trained network.


## 3. FCN Use in this Project

As a result, in this project, a FCN is trained to segment target from background in order for the drone to track it. Architecture of the trained FCN is shown in below figure.

[FCN]: ./docs/misc/FCN.png
![alt text][FCN]

### 3.1 Selected encoders

This FCN contains three encoders that each reduces the resolution by half and doubles the dimension:
- Encoder 1: filter_size=16, stride = 2
- Encoder 2: filter_size=32, stride = 2
- Encoder 3: filter_size=64, stride = 2

where the output of each encoder is:
- Encoder 1: (80, 80, 16)
- Encoder 2: (40, 40, 32)
- Encoder 3: (20, 20, 64)

Using more encoders can potentially extract more information of inputs however can drastically increase the size of the model.

The last encoder is a 1x1 convolution that has a dimension of 256, so the output is:
- 1x1 convolution: (20, 20, 256)

### 3.2 Selected decoders

In accordance with the encoders, 3 decoders are respectively designed to ensure the output eventually has the same size as the input image. Each decoder doubles the resolution of the input while reducing the dimension of the input by half. 

Since the encoders has substantially reduced the size of the input, many information that can be critical to reconstruct the output might be lost. Hence, 3 skip connections that link the outputs of encoders are introduced to decoders to provide more usefually information during the reconstruction. The output of the decoders are:
- Decoder 1: (40, 40, 64)
- Decoder 2: (80, 80, 32)
- Decoder 3: (160, 160, 16)

In the final layer, a 1x1 convolution with softmax activation is used to reduce the output to the dimension of 3 to identify each pixel among 3 classes, hence the final output's shape is (160, 160, 3)


## 4. Training

The training was done on a NVIDIA GPU and took about 2 hours to complete. The hyper parameters are:
- learning_rate = 0.1
- batch_size = 100
- num_epochs = 20
- steps_per_epoch = 100
- validation_steps = 50
- workers = 2

I started with `learning_rate = 0.01` and it turned out to be too small to converge. Eventually `0.1` worked OK. Then I fixed `batch_size = 100` and `steps_per_epoch = 100` and gradually increased `num_epochs` from `1` to `16` until the lose becomes relatively stable. I then increased `num_epochs` from `16` to `20` just to ensure lose is stable. I also increased `validation_steps` from `10` to `50` eventually to ensure each iteration is fully evaluated. `workers = 2` was used throughout the training since a GPU is used here.

[Training_Curve]: ./docs/misc/training_curve.png
![alt text][Training_Curve] 

The above figure shows the training curve of the above hyper parameters. In the future, more epochs can be used to further reduce the lose.

Eventually, the final IoU is `0.6028761565914768` and final score is `0.445740211311663`, which exceeds the project requirement of `0.4`.


## 5. Can Same Model Track Other Objects?

Yes, it can. We can keep the same FCN network, same training images with use new mask data to identify other objects. Since many features (object shape, color, etc.) in input image are already extracted in the trained FCN, we may also be able to use transfer learning to train the new model on top of the previously trained model to speed up.


## 6. Future Enhancements
- Run more epochs to have the model converge better.
- Use a bigger training data set to further improve the model.
- Try adding more layers in encoders and decoders to improve segmentation accuracy.
