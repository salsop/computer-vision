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