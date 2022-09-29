# My Computer Vision Experiment

I recently watched a [Build a restaurant edge solution with Google Cloud](https://www.youtube.com/watch?v=c2I4G7UH408) on YouTube, and was really interested in the [Dining Room Cleanliness Scenario](https://youtu.be/c2I4G7UH408?t=530) and wondered how hard it would be to create something that would be able to perform this Computer Vision use-case.

If you're interested in working though this with me, you'll need the following items:

* Raspberry PI v3b+
* USB WebCam
* Google Cloud Account
* Network Enabled Camera with RTSP suport.
* Coral USB TPU

Dealing with myu investigations in sequence,

1. Can I capture Video and Display this on the Screen.
2. Can I use TensorFlow Lite to do Object Identification on the Video Stream.
3. Can I train a customer TensorFlow model to recongnize something I may be interested in, such as a Dirty Table, or in my case a Dog Toy.
4. Can I use a Network Attached Video Camera instead of a USB WebCam to stream the video for the Object Identification Workflow.
5. Is it possible to make this Object Identification TensorFlow Model run faster with a connected TPU.

## Step 1 - Capturing a video stream from a USB webcam.

To run through this use-case I used Python code, and made use of the OpenCV module:


Here is the initial code that can be found in this repo [here](./code/step-1/video-capture.py)

```python
# import opencv2
import cv2

# open video capture device id 0
cap = cv2.VideoCapture(0)

# if video capture device cannot be opened print error and exit
if not cap.isOpened():
    print("error - cannot open video capture device")
    exit()

# Start Loop
while True:
    # capture frame from video capture device
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("error - cannot read frame")
        break

    # display the frame
    cv2.imshow('video', frame)

    # wait for ESC to be pressed, and exit loop if pressed
    if cv2.waitKey(1) == 27:
        break

# release the video capture device
cap.release()
cv2.destroyAllWindows()
```

## Step 2 - 