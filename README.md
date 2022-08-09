# Automatic Number Plate Recognition

> AI to detect and recognize number plates on vehicles.

## Table of contents

- [General information](#general-information)
- [Dataset](#dataset)
- [How does it work](#how-does-it-work)

## [Live demo](https://itsyoboieltr.github.io/anpr-ai/)

## General information

This is an AI that was trained on images of number plates to carry out number plate detection and recognition. It works for both image and videos. Video detection also includes object tracking.

<img width="300" src="https://user-images.githubusercontent.com/72046715/183776545-c51843c9-d350-4f4f-aa4f-1168e6922904.png">

## Dataset

For this project, I created the [ANPR dataset](https://archive.org/details/anpr-dataset), a dataset of approx. 30k handpicked images of number plates.

Annotations are in YOLOV7 format.

<img width="600" src="https://user-images.githubusercontent.com/72046715/183776762-7e0d9822-80a1-442e-a111-2fbc03b8213c.png">

## How does it work

Technologies used:

- [YoloV7](https://github.com/WongKinYiu/yolov7): Object detection model to detect the number plate
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR): OCR to read the number plate
- [Deep SORT](https://github.com/levan92/deep_sort_realtime): Object tracking algorithm for video detection

The YOLOV7 Model was fine-tuned using the ANPR dataset to detect number plates. When a number plate is detected, PaddleOCR is used to read the number plate. For video detection, Deep SORT is used to handle object tracking.

