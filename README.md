# HackOhio2022

## Welcome to our HackOhio project: De-Distracted Driving!!

We placed 1st overall and 2nd in the Honda challenge. Huge thank you to all the dedicated work done by the HackOhio team for putting on the event, and to Honda for being a dedicated sponsor.

<img
  src="https://user-images.githubusercontent.com/57300285/195107289-e1a78b87-c128-477b-9bc6-05a1c79be7fd.jpg"
  width="700">

This readme has the following sections: Overview, How to use files, Details of Implementation

## Overview

We used computer vision to determine if the driver of a car was distracted or not. 
If a driver was distracted we would light up red leds, sound a buzzer, and finally generate a vibration to alert the driver.
We mounted a webcam in front of the driver to detect if the driver is engaged or distracted. 

## How to use Files

### arduino.cpp

This file contains the code uploaded to the arduino. 
It allows us to control the leds, buzzer, and vibrate feature.

### hackohio2022.pptx

This is an overview of the project and contains 2 videos showcasing our work

### main.py

This is our main script. It runs inference on the video from the webcam.
If a driver is distracted it sends a signal to the arduino to activate alarms based on how long the driver is distracted.

### processVideo.py

This is a toy example used to verify that the methods used in process_video_into_training_data.py works correctly.

### process_video_into_training_data.py

Here we use a yolov5 model to automatically generate bounding boxes and labels for our drivers.

### train.yaml

Yaml file for training our model.

## Details of Implementation

### Data generation

We sat in a parking lot a generated approximatly 1 hour of videos. We had 4 'drivers', each driver recorded ~7.5 minutes of engaged and distracted driving.
Since we know that in our dataset the driver is always present we can use yolov5 to detect the driver. 
If in any frame no driver is detected, then we can throw that frame out, since no bounding box can be reliably generated. 
We sample every 8th frame of the videos to generate our dataset. Overall we generate and automatically annotate 13,037 frames. 

### Arduino
Here is an image of the curcuit/breadboard and arduino which we use to alarm the driver if they are distracted.
<img width="700" alt="Screen Shot 2022-10-11 at 3 02 27 PM" src="https://user-images.githubusercontent.com/57300285/195177059-b35a4aa6-c525-4f22-b222-240aedf52fa7.png">

