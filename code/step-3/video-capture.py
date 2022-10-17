# import opencv2
import cv2

# import tensorflow lite modules
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

# set variables for the tensorflow lite detection
model = "efficientdet_lite0.tflite"
enable_edgetpu = False
num_threads = 4

# open video capture device is now RTSP camera
cap = cv2.VideoCapture("rtsp://rtsp:12345678@192.168.1.96:554/av_stream/ch0")

# if video capture device cannot be opened print error and exit
if not cap.isOpened():
    print("error -> cannot open video capture device")
    exit()

# initialize the object detection model
base_options = core.BaseOptions(file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
detection_options = processor.DetectionOptions(max_results=10, score_threshold=0.3)
options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)

skipFramesCounter = 0

_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # green

# Start Loop
while True:
    # capture frame from video capture device
    skipFramesCounter += 1
    ret = cap.grab()

    # process every 10 frames
    if skipFramesCounter % 10 == 0:
        ret, frame = cap.retrieve()

        # if frame is read correctly ret is True
        if not ret:
            print("error -> cannot read frame")
            break

        # convert the image from BGR to RGB for the tensorflow lite model
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # create a tensorimage object from the RGB image
        input_tensor = vision.TensorImage.create_from_array(rgb_frame)

        # run object detection using the model
        detection_result = detector.detect(input_tensor)

        # draw boxes and labels on frame
        for detection in detection_result.detections:

            # draw bounding_box for detected object
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(frame, start_point, end_point, _TEXT_COLOR, 3)

            # draw label and score for detected object
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (_MARGIN + bbox.origin_x, _MARGIN + _ROW_SIZE + bbox.origin_y)
            cv2.putText(frame, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

        # display the frame full screen
        cv2.namedWindow("video", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('video', frame)

    # wait for ESC to be pressed, and exit loop if pressed
    if cv2.waitKey(1) == 27:
        break

# release the video capture device
cap.release()
cv2.destroyAllWindows()
