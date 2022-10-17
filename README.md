# My Computer Vision Experiment

I recently watched a [Build a restaurant edge solution with Google Cloud](https://www.youtube.com/watch?v=c2I4G7UH408) on YouTube, and was really interested in the [Dining Room Cleanliness Scenario](https://youtu.be/c2I4G7UH408?t=530) and wondered how hard it would be to create something that would be able to perform this Computer Vision use-case.

If you're interested in working though this with me, you'll need the following items:

* Debian Linux Machine with Camera (USB or Built-in)
* Google Cloud Account for the Model Training
* Coral USB TPU
* Network Enabled Camera with RTSP suport.

Dealing with myu investigations in sequence,

1. Can I capture Video and Display this on the Screen.
2. Can I use TensorFlow Lite to do Object Identification on the Video Stream.
3. Can I use a Network Attached Video Camera instead of a USB WebCam to stream the video for the Object Identification Workflow.
4. Can I train a customer TensorFlow model to recongnize something I may be interested in, such as a Dirty Table, or in my case a Dog Toy.
5. Is it possible to make this Object Identification TensorFlow Model run faster with a connected TPU.


## Step 0 - Prepare Debian Linux Machine

```
pip3 install tflite_support
pip3 install opencv-python
```

## Step 1 - Capturing a video stream from a USB webcam.

To run through this use-case I used Python code, and made use of the OpenCV module:

Here is the initial code that can be found in this repo [here](./code/step-1/video-capture.py)

## Step 2 - Object Identification on the video stream

To expand on the previous example, I included a TensorFlow lite model to analysing each frame, and drawing rectangles on the image identifying each object.

Download the model TensorFlow Lite model:
```shell
curl \
  -L 'https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1?lite-format=tflite' \
  -o 'efficientdet_lite0.tflite'
```

Here is the initial code that can be found in this repo [here](./code/step-2/video-capture.py)

## Step 3 - Object Identification on a video stream from a network attached camera

To test this out I ordered a cheap network attached webcam from Amazon that supports the RTSP protocol. This really could be any camera, but this was the one I found:

![Sonoff Wi-Fi Wireless IP Security Camera](./images/sonoff-camera.webp)

You then need to create the RTSP address which looks like this:

```
rtsp://rtsp:12345678@192.168.1.96:554/av_stream/ch0
```

The details from my camera are:

* Username: `rtsp`
* Password: `12345678`
* IP Address: `192.168.1.96`
* Port: `554`

The rest of the URL I had to obtain from searching the internet.

You can test this with [VLC Media Player](https://www.videolan.org/) by opening a Network and entering the URL. If its correct you should see the output from the Camera.

> My camera was reaching out to IP addresses on the internet, and as I didn't want to use any cloud related functions I prevented its access to the Internet by putting it behind a firewall.

Here is the code with now processing the RTSP stream in this repo [here](./code/step-3/video-capture.py)

## Step 4 - Does an Edge TPU help with performance?

I ordered a Coral USB TPU online from Amazon

![Coral USB](./images/coral-usb.webp)