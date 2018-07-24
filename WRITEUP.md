[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

# Deep Learning Project: Follow Me

In this project, a deep neural network is trained to identify and track a target in simulation. The model is then used with the simulator to track and follow the target.

[image_0]: ./docs/misc/sim_screenshot.png
![alt text][image_0] 

## I. Training the Neural Network

### 1. FCN

While a traditional Convolutional Neural Network (CNN) has a fully connected layer at the end, it cannot adapt to inputs with different sizes. On the contrary, a Fully Convolutional Network (FCN) only contains convolutional layers, hence can adapt to various input sizes. This benefit is due to convolution operation only takes a local area of the input at a time, therefore the overall size of the input can be arbitrary and the output is only determined by the shape of the filter. By changing the filter properly, the desired output can be acquired, which makes pixel by pixel semantic segmentation possible.

As a result, in this project, a FCN is trained to segment target from background in order for the drone to track it. Architecture of the trained FCN is shown in below figure.

[FCN]: ./docs/misc/FCN.png
![alt text][FCN] 

This FCN contains three encoders that each reduces the resolution by half and doubles the dimension:
- Encoder 1: filter_size=16, stride = 2
- Encoder 2: filter_size=32, stride = 2
- Encoder 3: filter_size=64, stride = 2

where the output of each encoder is:
- Encoder 1: (80, 80, 16)
- Encoder 2: (40, 40, 32)
- Encoder 3: (20, 20, 64)

A 1x1 convolution is used right after Encoder 3 that maintained the resolution but increases the dimension to 256, so the output is:
- 1x1 convolution: (20, 20, 256)

3 decoders are then used after the 1x1 convolution, where each of them doubles the resolution while reducing the dimension of the input by half. In addition, 3 skip connections are used to ensure the convergence of the model during training. The output of the decoders are:
- Decoder 1: (40, 40, 64)
- Decoder 2: (80, 80, 32)
- Decoder 3: (160, 160, 16)

In the end, a 1x1 convolution with softmax activation is used to identify each pixel among 3 classes, hence the final output is (160, 160, 3)


### 2. Training

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




