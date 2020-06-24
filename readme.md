# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

([Youtube link to watch the video](https://www.youtube.com/watch?v=b7DRvTuf9EU) )

[//]: # (Image References)

[center_after_bridge]: ./images/center_2020_06_23_22_15_12_106.jpg "Center after bridge"
[right_after_bridge]: ./images/right_2020_06_23_22_15_12_106.jpg "Right after bridge"
[left_after_bridge]: ./images/left_2020_06_23_22_15_12_106.jpg "Left after bridge"
[center_sharp]: ./images/center_2020_06_23_12_01_42_826.jpg "Center sharp"
[flip_sharp]: ./images/flipped.jpg "Flipped"
[flip_bridge]: ./images/flipped-bridge.jpg "Flipped"
[left_sharp]: ./images/left_2020_06_23_12_01_42_826.jpg "Left sharp"
[right_sharp]: ./images/right_2020_06_23_12_01_42_826.jpg "Right sharp"
[nvidia_model]: ./images/nvidia.png "Nvidia model"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* [Youtube link for one full lap record.](https://www.youtube.com/watch?v=b7DRvTuf9EU) 

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

I created a data generator which yields the training data and validation during model training. Initially, I suffered from insufficient memory error, and therefore, a data generator is very useful in this case.

```python 
def generate_data(data, batchSize = 32):
    while True:
        data = shuffle(data)
        for i in range(0, len(data), int(batchSize/4)):
            X_batch = []
            y_batch = []
            chunk = data[i: i+int(batchSize/4)]
            for line in chunk:
                center_image = cv2.imread(line[0])
                flipped_image = np.fliplr(center_image)
                left_image = cv2.imread(line[1])
                right_image = cv2.imread(line[2])
                angle = float(line[3])

                #center image
                X_batch.append(center_image)
                y_batch.append(angle)
                
                #flipped
                X_batch.append(flipped_image)
                y_batch.append(-angle)
                
                # left camera image and left-corrected angle
                X_batch.append(left_image)
                y_batch.append(angle + LEFT_ANGLE_CORRECTION)

                # right camera image and right-corrected angle
                X_batch.append(right_image)
                y_batch.append(angle + RIGHT_ANGLE_CORRECTION)
            # converting to numpy array
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            yield shuffle(X_batch, y_batch)
```



For each batch, data is fetched from the randomly shuffled dataset, each batch contains original center data, flipped data, left camera and right camera data with angle correction (0.4 for left image and - 0.3 for right image).
### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the NVIDIA CNN, which has six convolutional layers and 4 fully connected layers.

Normalization is also used to make sure the training process converges faster.

```python
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
```

The input is cut at top and down to remove irrelevant scenery and car components in the model.
```python
    model.add(Cropping2D(cropping=((50,20),(0,0))))
```

#### 2. Attempts to reduce overfitting in the model


I used sklearn to split the dataset into training and validation in the ratio of 80 : 20.

To reduce overfitting, the model was trained on the combination of multiple datasets. Datasets include one normal lap, and the repitition of left and right sharp turns.

I realised that in my model, using dropouts does not improve the general performance. However, I observed that the model frequently failed on the sharp turn, so I generated more data for these sharp turns, and found out that the model performed well after a large number of epoches on the sharp turn data.

To look specifically for the dataset used in training, please go to [Backup folder](./backup/)


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 91).

#### 4. Appropriate training data

Instead of using the dataset provided by Udacity, I used the my own dataset.

It included one normal lap of driving, and multiple repetitions of sharp turns, where the initial models failed the most.

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a simple CNN which has 3 convolution layers and 3 fully connected layers which aims to get myself familiarize with the simulator.
I created one-lap driving dataset however, the initial model failed to keep the car long near the road centre as it is too simple.

I decided to move to the NVIDIA model because it was widely accepted that the NVIDIA model is successful in the behavioral cloning task.

Using the one-lap data, the car could advance further but still failed to handle sharp turn.

I decided to give more data on these sharp turns and feed into the model. Also, I tried to increase the left and right angle correction to make sure that when the car was about to drive off the road center, it could turn back more aggressively. 

And the result improved as the car could now handle these sharp turns properly.

#### 2. Final Model Architecture

The final model architecture (model.py lines 77-89) consisted of a CNN with the following layers.

*    Input: array of 160x320x3
*    Normalize layer
*    Cropping layer
*    Convolutional layer using 24 5x5 filters, with RELU activation
*    Convolutional layer using 36 5x5 filters with RELU activation
*    Convolutional layer using 48 5x5 filters with RELU activation
*    Convolutional layer using 64 3x3 filters with RELU activation
*    Convolutional layer using 64 3x3 filters with RELU activation
*    Convolutional layer using 64 3x3 filters with RELU activation
*    Flatten layer
*    Fully connected layer with 100 neurons
*    Fully connected layer with 50 neurons
*    Fully connected layer with 10 neurons
*    Single neuron layer which is the result.


Here is a visualization of the architecture:

![Visualisation of NVIDIA model][nvidia_model]
#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one laps on track one using center lane driving. Here is an example image of center lane driving.

To augment the data set, I also flipped the image, with the associated angle being the opposite of the original angle. This increased the range of the dataset and helped the model generalize to drive in both direction.

Center image with an angle  `alpha`

![Center after bridge][center_after_bridge]
![Center sharp turn][center_sharp]

Flipped image with angle  `-alpha`

![Flipped image][flip_sharp]
![Flipped image bridge][flip_bridge]

To help the vehicle recovering from the left side and right sides of the road back to center, I used the left and right camera image, with aggressive correction towards the center.

Left image with corrected angle `alpha + 0.4`

![Left after bridge][left_after_bridge]
![Left sharp][left_sharp]

Right image with corrected angle `alpha - 0.3`

![Right after bridge][right_after_bridge]
![Right sharp][right_sharp]

### Reference
1. [Citlaligm's solution](https://github.com/citlaligm/Behavioral-cloning)
2. [Prathmesh Dali's solution](https://github.com/prathmesh-dali/CarND-term1-Behavioral-Cloning) 