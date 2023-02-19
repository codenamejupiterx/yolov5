# Face Detector Module (`FD`)

## Table of Content

1. [Introduction](#system-introduction)
2. [System Setup and Test Instructions](#system-setup-and-test-instructions)
3. [System Architecture](#system-architecture)
4. [System Output](#system-output)

## Introduction** **

The** ****Face Detector** is an object recognition module with a model that has been trained to detect instances human faces in images or video frames. At the heart of the** **`FD` module is a** **[YOLOv5](https://github.com/ultralytics/yolov5) Convolution Neural Network trained on custom datasets containing instances of faces of participants from a random television reality show. Examples of the dataset are listed below:

[![Sample Dataset](/msu-evrl/savia/raw/main/doc/images/docs-fd-sample-dataset.png)](/msu-evrl/savia/blob/main/doc/images/docs-fsd-sample-dataset.png)

## Setup and Test Instructions** **

1. Make sure to place the path to the video stream into the** **`src.fd.face_det_config.json` file under the** **`vid_path` key.
2. Ensure that you have the appropriate weights,** **`weights_path`, for the system to work.
3. Run the module** **`python -m src.fsd.main`


## System Architecture** **

The** **`FSD` reads in video stream then breaks them into frame and uses a custom trained YOLOv5 model to run fire and smoke inferences on each frame (see the diagram below).

## System Output


The outputs of the Fire and Smoke Detection module are:

1. Marked-up video frames with the the instances of fire and smoke in bounding boxes.
2. Console read outs of benchmarking stats and other pertinent information.
3. Creation of or appendings to a valid** **`EventStream` file in the root of the directory
